#%%
import time
import sys
import random
import csv
import numpy as np
from task import Task
from agents.agent import DDPG
from plot_functions import plot_results, plot_training_historic

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
            #rotor_speeds = [405]*4
            next_state, reward, done = task.step(rotor_speeds)

            to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(rotor_speeds) + [reward]
            for ii in range(len(labels)):
                results[labels[ii]].append(to_write[ii])
            writer.writerow(to_write)

            state = next_state
            if done:
                break

     # Restore noise
    agent.noise = OUNoise(agent.action_size, 
                         agent.exploration_mu, 
                         agent.exploration_theta, 
                         agent.exploration_sigma)

    print('Finished test episode!\n')
    return results

#%% Parameters
exploration_mu = 0
exploration_theta = 0.15
exploration_sigma = 0.2
buffer_size = 100000
batch_size = 64
gamma = 0.99
tau = 0.001
actor_learning_rate = 0.0001
critic_learning_rate = 0.001


#%% Training with agen
print('\n\nStart training...')
num_episodes = 500 # 1000
num_episodes_to_plot = max(100, num_episodes/5)
target_pos      = np.array([ 0.0, 0.0, 10.0])
init_pose       = np.array([ 0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
init_velocities = np.array([ 0.0, 0.0,  0.0])
task = Task(init_pose = init_pose,
           init_velocities = init_velocities,
           target_pos=target_pos)
agent = DDPG(task,
             exploration_mu =exploration_mu,
             exploration_theta = exploration_theta,
             exploration_sigma = exploration_sigma,
             buffer_size = buffer_size,
             batch_size = batch_size,
             gamma = gamma,
             tau = tau,
             actor_learning_rate = actor_learning_rate,
             critic_learning_rate = critic_learning_rate
             )

results = run_test_episode(agent, task, file_output)
plot_results(results, target_pos, 'Run without training')

#plt.show();import sys;sys.exit()
# Train
history = {'total_reward' : [], 'score' : [], 'i_episode' : []}
start = time.time()
done = False
for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new episode
    while True:
        action = agent.act(state) 
        next_state, reward, done = task.step(action)
        
        agent.step(action, reward, next_state, done)
        state = next_state

        if done:
            if len(history['i_episode'])>1:
                history['i_episode'].append(history['i_episode'][-1] + 1)
            else:
                history['i_episode'].append(1)
            history['total_reward'].append(agent.total_reward)
            history['score'].append(agent.score)
            print("\rEpisode = {: 4d}, score = {:7.3f}, total_reward = {:7.3f}".format(
                i_episode, agent.score, agent.total_reward), end="")
            break
    sys.stdout.flush()

    if i_episode%num_episodes_to_plot == 0:
        results = run_test_episode(agent, task, file_output)
        plot_results(results, target_pos, 'Run after training for {} episodes.'.format(i_episode), i_episode)


print('\nTime training: {:.1f} seconds\n'.format(time.time() - start))

plot_training_historic(history)

# the pose, velocity, and angular velocity of the quadcopter at the end of the episode
print(task.sim.pose)
print(task.sim.v)
print(task.sim.angular_v)
    
results = run_test_episode(agent, task, file_output)

plot_results(results, target_pos, 'Run after training for {} episodes.'.format(num_episodes))

plt.show(block=False)
input('Press enter to finish')


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

