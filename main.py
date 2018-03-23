#%%
import time
import sys
import os
import csv
import numpy as np
from task import Task
from agents.agent import DDPG
from plot_functions import plot_results, plot_training_historic
from collections import defaultdict
import copy

import matplotlib.pyplot as plt
from agents.ou_noise import OUNoise

## Modify the values below to give the quadcopter a different starting position.
file_output = 'data.txt'                         # file name for saved results
plt.close('all')


# Run task with agent
def run_test_episode(agent : DDPG, task : Task, file_output):
    print('\nRunning test episode ...')

    labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
              'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
              'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4','reward']
    results = {x : [] for x in labels}

    aux_noise = copy.copy(agent.noise)
    agent.noise = OUNoise(agent.action_size, 0.0, 0.0, 0.0)
    
    state = agent.reset_episode() # start a new episode
    rewards_lists = defaultdict(list)
    print('state', state)
    print('state.shape', state.shape)
    
    # Run the simulation, and save the results.
    with open(file_output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)
        while True:
            rotor_speed = agent.act(state)
            rotor_speeds = np.array([rotor_speed]*4)
            #rotor_speeds = [405]*4
            # rotor_speeds = [500, 490, 500, 500]
            next_state, reward, done, new_rewards = task.step(rotor_speeds)
            for key, value in new_rewards.items():
                rewards_lists[key].append(value)

            to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(rotor_speeds) + [reward]
            for ii in range(len(labels)):
                results[labels[ii]].append(to_write[ii])
            writer.writerow(to_write)

            state = next_state
            if done:
                break

     # Restore noise
    agent.noise = copy.copy(aux_noise)

    print('Finished test episode!\n')
    return results, rewards_lists

#%% Parameters
class Params:
    pass
params = Params()
params.extra_text = 'with_surviving_reward__batch_norm'
params.exploration_mu = 0
params.exploration_theta = 0.15
params.exploration_sigma = 0.05
params.actor_learning_rate = 1.0e-5 # 0.0001
params.critic_learning_rate = 0.001  # 0.001
params.tau = 0.001 # 0.001
params.actor_net_cells = [16, 16]
params.critic_net_cells = [16, 32]

gamma = 0.99
buffer_size = 100000
batch_size = 64

num_episodes = 1000 # 1000

#%% Training with agen
print('\n\nStart training...')
num_episodes_to_plot = max(40, num_episodes/5)
target_pos      = np.array([ 0.0, 0.0, 10.0])
init_pose       = np.array([ 0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
init_velocities = np.array([ 0.0, 0.0,  0.0])
task = Task(init_pose = init_pose,
           init_velocities = init_velocities,
           target_pos=target_pos)
agent = DDPG(task,
             exploration_mu = params.exploration_mu,
             exploration_theta = params.exploration_theta,
             exploration_sigma = params.exploration_sigma,
             buffer_size = buffer_size,
             batch_size = batch_size,
             gamma = gamma,
             tau = params.tau,
             actor_learning_rate = params.actor_learning_rate,
             critic_learning_rate = params.critic_learning_rate,
             actor_net_cells = params.actor_net_cells,
             critic_net_cells = params.critic_net_cells
             )

results, rewards_lists = run_test_episode(agent, task, file_output)
plot_results(results, target_pos, 'Run without training', rewards_lists, num = 0, params = params)

# plt.show();import sys;sys.exit()

# Train
max_reward = -np.inf
last_i_max_reward = 300
history = {'total_reward' : [], 'score' : [], 'i_episode' : []}
start = time.time()
done = False
i_episode = 1
stuck_counter = 0
while i_episode < num_episodes+1:
    state = agent.reset_episode() # start a new episode
    t_episode = 0
    time_step_episode = 0
    cum_sum_actions = np.array([0.0]*4)

    start = time.time()
    while True:
        action = agent.act(state, (i_episode-1)%100 == 0)
        actions = np.array(action*4)
        next_state, reward, done, _ = task.step(actions)

        agent.step(actions, reward, next_state, done)
        state = next_state

        cum_sum_actions += actions
        time_step_episode += 1
        #
        # print(
        #     "\r  Ep:{: 4d}. Step:{: 4d} (stuck:{: 5d}), reward: {:8.2f}, noise(sigma: {:6.3f}, theta: {:6.3f}, state, {:6.1f})(action:{:6.1f}), done: {}".
        #     format(i_episode,
        #            time_step_episode,
        #            stuck_counter,
        #            reward,
        #            agent.noise.sigma,
        #            agent.noise.theta,
        #            agent.noise.state[0],
        #            action[0],
        #            done
        #            ),
        #     end = ''
        # )

        if action[0] is np.nan:
            import sys
            sys.exit()

        t_episode += 0.06 # each step is 3 times 20 ms (50Hz)
        if done:

            episode_time = time.time() - start
            start = time.time()
            i_episode += 1

            # Slowly decrease noise if everything goes all right.
            # agent.noise.sigma = params.exploration_sigma/i_episode
            # agent.noise.theta = params.exploration_theta/i_episode

            if len(history['i_episode'])>1:
                history['i_episode'].append(history['i_episode'][-1] + 1)
            else:
                history['i_episode'].append(1)
            history['total_reward'].append(agent.total_reward)
            history['score'].append(agent.score)


            print("\rEpisode:{: 4d} (stuck:{: 5d}), score: {:7.1f}, reward: {:8.2f}, noise(sigma: {:6.3f}, theta: {:6.3f}, state, {:6.1f})(action:{:6.1f}). Time: {:3.0f} s.".
                  format(i_episode,
                         stuck_counter,
                         agent.score,
                         agent.total_reward,
                         agent.noise.sigma,
                         agent.noise.theta,
                         *[rotor for rotor in agent.noise.state],
                         *[cum_sum_action/time_step_episode for cum_sum_action in cum_sum_actions],
                         episode_time
                         ),
                  end="")

            break
    sys.stdout.flush()

    if i_episode%num_episodes_to_plot == 0 and stuck_counter == 0:
        results, rewards_lists = run_test_episode(agent, task, file_output)
        plot_results(results,
                     target_pos,
                     'Run after training for {} episodes.'.format(i_episode),
                     rewards_lists,
                     num = i_episode,
                     params = params)
    if (max_reward < reward) and (last_i_max_reward + 20 < i_episode):
        results, rewards_lists = run_test_episode(agent, task, file_output)
        plot_results(results,
                     target_pos,
                     'New max at: {} episodes.'.format(i_episode),
                     rewards_lists,
                     i_episode,
                     params =params)
        max_reward = reward
        last_i_max_reward = i_episode
        plt.close()

print('\nTime training: {:.1f} seconds\n'.format(time.time() - start))

plot_training_historic(history, params)

# the pose, velocity, and angular velocity of the quadcopter at the end of the episode
print(task.sim.pose)
print(task.sim.v)
print(task.sim.angular_v)
    
results, rewards_lists = run_test_episode(agent, task, file_output)

plot_results(results,
             target_pos,
             'Run after training for {} episodes.'.format(num_episodes),
             rewards_lists,
             num = 0,
             params = params)

plt.show(block=False)


#%%
#plt.figure()
#plt.plot(results['time'], results['x_velocity'], label='x_hat')
#plt.plot(results['time'], results['y_velocity'], label='y_hat')
#plt.plot(results['time'], results['z_velocity'], label='z_hat')
#plt.legend()
#_ = plt.ylim()
#plt.show(block=False)
#
#
#plt.legend()
#_ = plt.ylim()
#plt.show(block=False)
#
#
#plt.figure()
#plt.plot(results['time'], results['phi_velocity'], label='phi_velocity')
#plt.plot(results['time'], results['theta_velocity'], label='theta_velocity')
#plt.plot(results['time'], results['psi_velocity'], label='psi_velocity')
#plt.legend()
#_ = plt.ylim()
#plt.show(block=False)




#%%

