import matplotlib.pyplot as plt
import numpy as np


def plot_training_results(episodes_rewards, training_rewards, training_actor_losses, training_critic_losses, interval=100, show_plot = False, save_plot = True):
    """
    Plots the average rewards per specified number of episodes.

    Parameters:
    - episodes_rewards: list or array of episodic rewards.
    - interval: the number of episodes per average (default is 100).
    """
    plt.figure(figsize=(15, 10))
    #print(episodes_rewards)
    print(training_rewards)
    print(training_actor_losses)
    print(training_critic_losses)
    # Compute the number of intervals and average rewards for plotting
    num_intervals = len(episodes_rewards) // interval
    average_rewards = [np.mean(episodes_rewards[i * interval:(i + 1) * interval]) for i in range(num_intervals)]
    x_labels = [(i + 1) * interval for i in range(num_intervals)]

    # Plotting average rewards
    plt.subplot(2, 2, 1)
    plt.plot(x_labels, average_rewards, marker='o', linestyle='-')
    plt.title('Average Rewards per {} Episodes'.format(interval))
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')

    # Convert actor losses from tensors to a list of scalars
    actor_losses_scalar = [loss[0].item() for loss in training_actor_losses]

    # Plotting actor losses
    plt.subplot(2, 2, 2)
    plt.plot(actor_losses_scalar, label='Actor Loss', color='red')
    plt.title('Actor Losses Per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting critic losses
    # Convert critic losses from tensors to a list of scalars
    critic_losses_scalar = [loss[0].item() for loss in training_critic_losses]

    # Plotting critic losses
    plt.subplot(2, 2, 3)
    plt.plot(critic_losses_scalar, label='Critic Loss', color='blue')
    plt.title('Critic Losses Per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    """
    # Convert training rewards from tensors to a list of scalars
    training_rewards_scalar = [reward[0].item() for reward in training_rewards]

    # Plotting training rewards for each episode
    plt.subplot(2, 2, 4)
    plt.plot(training_rewards_scalar, label='Training Rewards', color='green')
    plt.title('Training Rewards Per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    """

    plt.tight_layout()

    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig('training_results.png', dpi=300)
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
