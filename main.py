#%%

import sys
import random
import csv
import numpy as np
from task import Task

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from agents.ou_noise import OUNoise

## Modify the values below to give the quadcopter a different starting position.
#runtime = 5.                                     # time limit of the episode
#init_pose = np.array([0., 0., 10., 0., 0., 0.])  # initial pose
#init_velocities = np.array([0., 0., 0.])         # initial velocities
#init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities
file_output = 'data.txt'                         # file name for saved results

# Setup
#task = Task(init_pose, init_velocities, init_angle_velocities, runtime)

# Run task with agent
def run_episode(agent, task, file_output):
    done = False
    labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
              'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
              'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
    results = {x : [] for x in labels}
    
       
    
    agent.noise = OUNoise(agent.action_size, 0.0, 0.0, 0.0)
    
    state = agent.reset_episode() # start a new episode
    print('state', state)
    print('state.shape', state.shape)
    
    # Run the simulation, and save the results.
    with open(file_output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)
        while True:
            rotor_speeds = agent.act(state)
            _, _, done = task.step(rotor_speeds)
            to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(rotor_speeds)
            for ii in range(len(labels)):
                results[labels[ii]].append(to_write[ii])
            writer.writerow(to_write)
            if done:
                break
            
    # the pose, velocity, and angular velocity of the quadcopter at the end of the episode
    print(task.sim.pose)
    print(task.sim.v)
    print(task.sim.angular_v)
    
     # Noise process
    agent.noise = OUNoise(agent.action_size, 
                         agent.exploration_mu, 
                         agent.exploration_theta, 
                         agent.exploration_sigma)

    return results

def plot_results(results, target_pos):
    
    #%matplotlib inline
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
    ax1.plot(results['time'], results['x'], label='x')
    ax1.plot(results['time'], results['y'], label='y')
    ax1.plot(results['time'], results['z'], label='z')
    ax1.legend()
    
    ax2.plot(results['time'], results['rotor_speed1'], label='Rotor 1 revolutions / second')
    ax2.plot(results['time'], results['rotor_speed2'], label='Rotor 2 revolutions / second')
    ax2.plot(results['time'], results['rotor_speed3'], label='Rotor 3 revolutions / second')
    ax2.plot(results['time'], results['rotor_speed4'], label='Rotor 4 revolutions / second')
    ax2.legend()
    plt.show()  
    
    fig = plt.figure(figsize=(12,6))
    ax = fig.gca(projection='3d')
    ax.plot(results['x'], results['y'], results['z'], label='parametric curve')
    ax.plot([target_pos[0]], [target_pos[1]], [target_pos[2]], 'ro', markersize=12, label='target')
    ax.plot([results['x'][0]],  [results['y'][0]],  [results['z'][0]], 'gx', markersize=6, label='start')
    ax.plot([results['x'][-1]], [results['y'][-1]], [results['z'][-1]], 'bx', markersize=6, label='end')
    ax.legend()
    
    plt.show()

#%% Training with agen
from agents.agent import DDPG

num_episodes = 15000 # 1000
target_pos      = np.array([ 0.0, 0.0, 10.0])
init_pose       = np.array([10.0, 0.0,  0.0, 0.0, 0.0, 0.0])
init_velocities = np.array([ 0.0, 0.0,  0.0])
#task = Task(init_pose = init_pose,
#            init_velocities = init_velocities,
#            target_pos=target_pos)
task = Task(target_pos=target_pos)
agent = DDPG(task) 

results = run_episode(agent, task, file_output)
plot_results(results, target_pos)
 
# Train
history = {'total_reward' : [], 'score' : [], 'i_episode' : []}
for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new episode
    while True:
        action = agent.act(state) 
        next_state, reward, done = task.step(action)
        
        agent.step(action, reward, next_state, done)
        state = next_state
        if done:
            history['i_episode'].append(i_episode)
            history['total_reward'].append(agent.total_reward)
            history['score'].append(agent.score)
            print("\rEpisode = {: 4d}, score = {:7.3f}, total_reward = {:7.3f}".format(
                i_episode, agent.score, agent.total_reward), end="")
            break
    sys.stdout.flush()
    
f, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
ax1.plot(history['i_episode'], history['total_reward'], label='total_reward')
ax1.set_ylim([min(history['total_reward'])/10.0, max(history['total_reward'])])
ax1.legend()

ax2.plot(history['i_episode'], history['score'], label='score')
ax2.set_ylim([min(history['score'])/10.0, max(history['score'])])
ax2.legend()
plt.show()  
# the pose, velocity, and angular velocity of the quadcopter at the end of the episode
print(task.sim.pose)
print(task.sim.v)
print(task.sim.angular_v)
    
results = run_episode(agent, task, file_output)

plot_results(results, target_pos)

#%%
#plt.figure()
#plt.plot(results['time'], results['x_velocity'], label='x_hat')
#plt.plot(results['time'], results['y_velocity'], label='y_hat')
#plt.plot(results['time'], results['z_velocity'], label='z_hat')
#plt.legend()
#_ = plt.ylim()
#plt.show()
#
#
#plt.figure()
#plt.plot(results['time'], results['phi'], label='phi')
#plt.plot(results['time'], results['theta'], label='theta')
#plt.plot(results['time'], results['psi'], label='psi')
#plt.legend()
#_ = plt.ylim()
#plt.show()
#
#
#plt.figure()
#plt.plot(results['time'], results['phi_velocity'], label='phi_velocity')
#plt.plot(results['time'], results['theta_velocity'], label='theta_velocity')
#plt.plot(results['time'], results['psi_velocity'], label='psi_velocity')
#plt.legend()
#_ = plt.ylim()
#plt.show()




#%%

