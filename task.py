import numpy as np
from physics_sim import PhysicsSim
from collections import defaultdict


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self,
                 init_pose = None,
                 init_velocities = None,
                 init_angle_velocities = None,
                 runtime = 5.,
                 target_pos = None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6

        self.action_low = 50
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def manhattan_distance(self, point_a, point_b):

        return (abs(point_a - point_b)).sum()

    def euclidean_distance(self, point_a, point_b):

        return np.linalg.norm(point_a - point_b)

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        rewards = defaultdict(float)
        rewards['surviving'] = 10.0/50.0 # plus 10 per second surviving at 50 Hz

        # reward = 1.0 - 0.3 * self.euclidean_distance(self.sim.pose[:3], self.target_pos)
        # reward = 1.0 - 0.3 * self.manhattan_distance(self.sim.pose[:3], self.target_pos)

        rewards['distance'] = - 1.0 * self.euclidean_distance(self.sim.pose[:3], self.target_pos)

        rewards['angles'] = -np.abs(self.sim.pose[3:6]).sum() * 18/np.pi
      #  reward += -np.abs(self.sim.angular_v).sum() * 90/np.pi

        reward = sum([x for x in rewards.values()])
        return reward, rewards

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        rewards = defaultdict(float)
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            instant_reward, new_rewards = self.get_reward()
            reward += instant_reward
            for key, value in new_rewards.items():
                rewards[key] += value
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done, rewards

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state