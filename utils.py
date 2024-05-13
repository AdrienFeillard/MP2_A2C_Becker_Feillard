import matplotlib.pyplot as plt
import numpy as np


def plot_average_rewards(episodes_rewards, interval=100, show_plot = False, save_plot = True):
    """
    Plots the average rewards per specified number of episodes.

    Parameters:
    - episodes_rewards: list or array of episodic rewards.
    - interval: the number of episodes per average (default is 100).
    """
    # Compute the number of intervals
    num_intervals = len(episodes_rewards) // interval

    # Calculate average rewards for each interval
    average_rewards = [np.mean(episodes_rewards[i * interval:(i + 1) * interval]) for i in range(num_intervals)]

    # Prepare x-axis labels
    x_labels = [(i + 1) * interval for i in range(num_intervals)]

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(x_labels, average_rewards, marker='o', linestyle='-')
    plt.title('Average Rewards per {} Episodes'.format(interval))
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.show()
    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig(f'rewards_plot_through_all_training.png')
def plot_critic_values(states, values, K, n_steps, n_iteration, save=False, display=True):
    """fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].plot(states[:, 0], values, 'b.')
    axs[0, 0].set_title('Cart Position vs Value')
    axs[0, 0].set_xlabel('Cart Position')
    axs[0, 0].set_ylabel('Critic Value')

    axs[0, 1].plot(states[:, 1], values, 'g.')
    axs[0, 1].set_title('Cart Velocity vs Value')
    axs[0, 1].set_xlabel('Cart Velocity')
    axs[0, 1].set_ylabel('Critic Value')

    axs[1, 0].plot(states[:, 2], values, 'r.')
    axs[1, 0].set_title('Pole Angle vs Value')
    axs[1, 0].set_xlabel('Pole Angle')
    axs[1, 0].set_ylabel('Critic Value')

    axs[1, 1].plot(states[:, 3], values, 'm.')
    axs[1, 1].set_title('Pole Angular Velocity vs Value')
    axs[1, 1].set_xlabel('Pole Angular Velocity')
    axs[1, 1].set_ylabel('Critic Value')

    plt.tight_layout()"""

    time_steps = range(len(values))

    plt.figure(figsize=(10, 5))  # Adjust the figure size as needed

    plt.plot(time_steps, values, label='Value Function')

    plt.title('Value Function Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('State-Value Estimate')
    plt.legend()
    plt.grid(True)

    if save:
        filename = f'critic_values_K{K}_steps{n_steps}_iter{n_iteration}.png'
        plt.savefig(filename)
        print(f"Plot saved as {filename}")

    # Display the plot if the display flag is True
    if display:
        plt.show()
    else:
        plt.close()
