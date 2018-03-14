
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_results(results, target_pos, title = ''):
    # %matplotlib inline

    fig = plt.figure(figsize=(12, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
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

    plt.suptitle(title)

    plt.show(block=False)


def plot_training_historic(history):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(history['i_episode'], history['total_reward'], label='total_reward')
    ax1.set_ylim([min(history['total_reward']) / 10.0, max(history['total_reward'])])
    ax1.legend()

    ax2.plot(history['i_episode'], history['score'], label='score')
    ax2.set_ylim([min(history['score']) / 10.0, max(history['score'])])
    ax2.legend()
    plt.show(block=False)
