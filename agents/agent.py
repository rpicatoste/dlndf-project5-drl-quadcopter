import numpy as np

from task import Task
from .actor import Actor
from .critic import Critic
from .ou_noise import OUNoise
from .replay_buffer import ReplayBuffer


class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self,
                 task : Task,
                 params,
                 buffer_size = 100000,
                 batch_size = 64
    ):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size,
                                 self.action_size,
                                 self.action_low,
                                 self.action_high,
                                 net_cells_list = params.actor_net_cells,
                                 learning_rate = params.actor_learning_rate)
        self.actor_target = Actor(self.state_size,
                                  self.action_size,
                                  self.action_low,
                                  self.action_high,
                                  net_cells_list = params.actor_net_cells,
                                  learning_rate = params.actor_learning_rate)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size,
                                   self.action_size,
                                   net_cells = params.critic_net_cells,
                                   learning_rate = params.critic_learning_rate)
        self.critic_target = Critic(self.state_size,
                                    self.action_size,
                                    net_cells = params.critic_net_cells,
                                    learning_rate = params.critic_learning_rate)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = params.exploration_mu
        self.exploration_theta = params.exploration_theta
        self.exploration_sigma = params.exploration_sigma
        self.noise = OUNoise(self.action_size, 
                             self.exploration_mu,
                             self.exploration_theta,
                             self.exploration_sigma)

        # Replay memory
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = params.gamma  # discount factor
        self.tau = params.tau  # for soft update of target parameters

        # Reward and score
        self.score = 0
        self.total_reward = 0
        self.count = 0

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.total_reward = 0
        self.count = 0
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)
        # print('  Memory size: {}, vals: '.format(len(self.memory)))
        # print('    - last_pos {}'.format(self.last_state[:3]))
        # print('    - next_pos: {}'.format(next_state[:3]))
        # print('    - last_ang {}'.format(self.last_state[3:6]))
        # print('    - next_ang: {}'.format(next_state[3:6]))
        # print('    - action: {}'.format(action))
        # print('    - reward: {}'.format(reward))
        # print('    - done: {}'.format(done))
        # store rewards
        self.total_reward += reward
        self.count += 1      
        self.score = self.total_reward / float(self.count) if self.count else 0.0

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, states, trick = False):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(states, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]

        if trick:
            action = [404] * self.action_size
        # print('  - act - states - ', states[:6])
        # print('  - act - action 1               - {:.2f}'.format(action[0]))
        noise = self.noise.sample()
        # print('  - act - list(action + noise) 1 - {:.2f}'.format(list(action + noise)[0]))

        return list(action + noise)  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
                
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, 4)#self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # print('  Learning from {} experiences! (received {})'.format(len(states), len(experiences)))
        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        action_next = self.actor_target.model.predict_on_batch(next_states)
        actions_next = np.array(action_next*4)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions[:,0]], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions[:,0].reshape(-1,1), 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)