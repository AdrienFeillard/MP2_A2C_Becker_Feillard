import matplotlib.pyplot as plt

# Function to plot the training results
"""def plot_training_results():
    plt.figure(figsize=(12, 5))

    # Plotting episode rewards
    plt.subplot(1, 3, 1)
    plt.plot(all_episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    # Plotting actor losses
    plt.subplot(1, 3, 2)
    plt.plot(actor_losses)
    plt.title('Actor Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')

    # Plotting critic losses
    plt.subplot(1, 3, 3)
    plt.plot(critic_losses)
    plt.title('Critic Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()"""


# Visualize the training results
# plot_training_results()

def plot_critic_values(states, values):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
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

    plt.tight_layout()
    plt.show()
