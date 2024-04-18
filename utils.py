import matplotlib.pyplot as plt
# Function to plot the training results
def plot_training_results():
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
    plt.show()

# Visualize the training results
plot_training_results()