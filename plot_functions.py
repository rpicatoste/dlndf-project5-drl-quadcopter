
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_results(results, target_pos, title = '', num = 0):
    # %matplotlib inline

    fig = plt.figure(figsize=(12, 10))

    ax1 = fig.add_subplot(3, 2, 1)
    ax3 = fig.add_subplot(3, 2, 3)
    ax5 = fig.add_subplot(3, 2, 5)

    ax2 = fig.add_subplot(2, 2, 2)
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')

    ax1.plot(results['time'], results['x'], label='x')
    ax1.plot(results['time'], results['y'], label='y')
    ax1.plot(results['time'], results['z'], label='z')
    ax1.legend()
    ax1.grid()

    ax2.plot(results['time'], results['rotor_speed1'], label='Rotor 1 revolutions / second')
    ax2.plot(results['time'], results['rotor_speed2'], label='Rotor 2 revolutions / second')
    ax2.plot(results['time'], results['rotor_speed3'], label='Rotor 3 revolutions / second')
    ax2.plot(results['time'], results['rotor_speed4'], label='Rotor 4 revolutions / second')
    ax2.legend()
    ax2.grid()

    ax3.plot(results['time'], results['reward'], label='reward')
    ax3.legend()
    ax3.grid()

    ax4.plot(results['x'], results['y'], results['z'], label='parametric curve')
    ax4.plot([target_pos[0]], [target_pos[1]], [target_pos[2]], 'ro', markersize=12, label='target')
    ax4.plot([results['x'][0]], [results['y'][0]], [results['z'][0]], 'gx', markersize=6, label='start')
    ax4.plot([results['x'][-1]], [results['y'][-1]], [results['z'][-1]], 'bx', markersize=6, label='end')
    ax4.legend()

    ax4.set_xlabel('X axis')
    ax4.set_ylabel('Y axis')
    ax4.set_zlabel('Z axis')


    ax5.plot(results['time'], results['phi'], label='phi')
    ax5.plot(results['time'], results['theta'], label='theta')
    ax5.plot(results['time'], results['psi'], label='psi')
    ax5.legend()

    plt.suptitle(title)

    try:
        plt.savefig(r'figures\last_fig_plot_results' + str(num) + '.png')
    except:
        print('Image open or something, could not save.')


def plot_training_historic(history):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    N = 20
    total_reward = np.array(history['total_reward'])
    total_reward = np.convolve(total_reward, np.ones((N,)) / N, mode='valid')
    total_reward = np.concatenate([np.array([total_reward[0]]*(N-1)), total_reward])

    score = np.array(history['score'])
    score = np.convolve(score, np.ones((N,)) / N, mode='valid')
    score = np.concatenate([np.array([score[0]]*(N-1)), score])

    ax1.plot(history['i_episode'], history['total_reward'], label='total_reward')
    ax1.plot(history['i_episode'], total_reward, label='total_reward_filt')
    ax1.set_ylim([min(total_reward), max(total_reward)])
    ax1.legend()

    ax2.plot(history['i_episode'], history['score'], label='score')
    ax2.plot(history['i_episode'], score, label='score_filt')
    ax2.set_ylim([min(score), max(score)])
    ax2.legend()

    try:
        plt.savefig(r'figures\results.png')
    except:
        print('Image open or something, could not save.')
