import os

import matplotlib.pyplot as plt
import numpy as np


def plot_training_results(, save=False, display=True):

    time_steps = range(len(values))

    plt.figure(figsize=(10, 5))  # Adjust the figure size as needed

    plt.plot(time_steps, values, label='Value Function')

    plt.title('Value Function Over Trajectory')
    plt.xlabel('Time Steps')
    plt.ylabel('State-Value Estimate')
    plt.legend()
    plt.grid(True)

    path = f'plots/values_over_trajectory/seed_{seed}/actor_{actor}'
    os.makedirs(path, exist_ok=True)

    if save:
        filename = f'{path}/iter_{n_iteration}.png'
        plt.savefig(filename)
        print(f"Plot saved as {filename}")

    # Display the plot if the display flag is True
    if display:
        plt.show()
    else:
        plt.close()


def plot_values_over_trajectory(seed, values, actor, n_iteration, save=False, display=True):
    time_steps = range(len(values))

    plt.figure(figsize=(10, 5))  # Adjust the figure size as needed

    plt.plot(time_steps, values, label='Value Function')

    plt.title('Value Function Over Trajectory')
    plt.xlabel('Time Steps')
    plt.ylabel('State-Value Estimate')
    plt.legend()
    plt.grid(True)

    path = f'plots/values_over_trajectory/seed_{seed}/actor_{actor}'
    os.makedirs(path, exist_ok=True)

    if save:
        filename = f'{path}/iter_{n_iteration}.png'
        plt.savefig(filename)
        print(f"Plot saved as {filename}")

    # Display the plot if the display flag is True
    if display:
        plt.show()
    else:
        plt.close()
