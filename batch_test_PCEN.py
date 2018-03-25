#%%
import numpy as np
from agents.agent import DDPG
from task import Task
from helpers import Params, run_training


## Modify the values below to give the quadcopter a different starting position.
file_output = 'data.txt'                         # file name for saved results

buffer_size = 100000
batch_size = 64

num_episodes = 1000 # 1000

print('\n\nStart training...')
target_pos      = np.array([ 0.0, 0.0, 10.0])
init_pose       = np.array([ 0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
init_velocities = np.array([ 0.0, 0.0,  0.0])


params = Params()
params.extra_text = 'speed_reward_multip__concatenate'
params.exploration_mu = 0
params.exploration_theta = 0.15
params.exploration_sigma = 0.02 #0.002
params.actor_learning_rate = 1.0e-5 # 0.0001
params.critic_learning_rate = 0.001  # 0.001
params.tau = 0.01
params.actor_net_cells = [16*2, 16*2]
params.critic_net_cells = [16*2, 32*2]
params.gamma = 0.99

test_values = [1.0e-3, 1.0e-4, 1.0e-5,1.0e-6, 1.0e-7] # actor_learning_rate
# test_values = [1.0e-2, 1.0e-3, 1.0e-4,1.0e-5] # critic_learning_rate
# test_values = [0.9, 0.99] # gamma
# test_values = [0.2, 0.02, 0.002, 0.0002] # exploration_sigma
# test_values = [0.1, 0.01, 0.001, 0.0001] # tau
# test_values = [0.9, 0.99] # gamma
# Think how to do the networks batch.

for test_value in test_values:
    params.actor_learning_rate = test_value

    task = Task(init_pose = init_pose,
                init_velocities = init_velocities,
                target_pos = target_pos)
    agent = DDPG(task,
                 params,
                 buffer_size = buffer_size,
                 batch_size = batch_size
                 )

    run_training(agent, task, params, num_episodes, file_output)