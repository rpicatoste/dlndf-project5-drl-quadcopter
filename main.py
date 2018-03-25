#%%
import numpy as np
from agents.agent import DDPG
from task import Task
from helpers import Params, run_training


## Modify the values below to give the quadcopter a different starting position.
file_output = 'data.txt'                         # file name for saved results


params = Params()
params.extra_text = 'speed_reward_multiplier_gaussian_init_concatenate_fixing_sim'
params.exploration_mu = 0
params.exploration_theta = 0.15
params.exploration_sigma = 0.02 #0.002
params.actor_learning_rate = 1.0e-5 # 0.0001
params.critic_learning_rate = 0.001  # 0.001
params.tau = 0.01
params.actor_net_cells = [16*2, 16*2]
params.critic_net_cells = [16*2, 32*2]
params.gamma = 0.99

# random other values
# params.extra_text = 'speed_reward_multiplier_gaussian_init_concatenate_fixing_sim'
# params.exploration_mu = 0
# params.exploration_theta = 0.15
# params.exploration_sigma = 0.2 #0.002
# params.actor_learning_rate = 0.005 #1.0e-5 # 0.0001
# params.critic_learning_rate = 0.005 #0.001  # 0.001
# params.tau = 0.01 # 0.001
# params.actor_net_cells = [16*2, 16*2]
# params.critic_net_cells = [16*2, 32*2]
# gamma = 0.9#0.99

buffer_size = 100000
batch_size = 64

num_episodes = 1000 # 1000

print('\n\nStart training...')
target_pos      = np.array([ 0.0, 0.0, 10.0])
init_pose       = np.array([ 0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
init_velocities = np.array([ 0.0, 0.0,  0.0])

task = Task(init_pose = init_pose,
            init_velocities = init_velocities,
            target_pos = target_pos)
agent = DDPG(task,
             params,
             buffer_size = buffer_size,
             batch_size = batch_size
             )

run_training(agent, task, params, num_episodes, file_output)