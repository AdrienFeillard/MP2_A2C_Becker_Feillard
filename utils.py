import os

import matplotlib.pyplot as plt
import numpy as np


def exponential_moving_average(data, alpha=0.3):
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t - 1]
    return ema


def aggregate_plot(iterations, values, ylabel, title, name, log_scale=False, symlog_scale=False):
    assert not log_scale or not symlog_scale, "Cannot use both log and symlog scale at the same time"

    plt.figure(figsize=(10, 5))  # Adjust the figure size as needed

    min_values = np.min(values, axis=0)
    max_values = np.max(values, axis=0)
    avg_values = np.mean(values, axis=0)

    # Apply smoothing
    alpha = 0.1  # Smoothing factor
    smoothed_min_values = exponential_moving_average(min_values, alpha)
    smoothed_max_values = exponential_moving_average(max_values, alpha)
    smoothed_avg_values = exponential_moving_average(avg_values, alpha)

    adjusted_iterations = iterations[:len(smoothed_avg_values)]

    plt.fill_between(iterations, min_values, max_values, alpha=0.3, color='lightblue', label='Min-Max Range (Raw)')
    plt.plot(iterations, avg_values, label='Average (Raw)', color='blue', linestyle='dotted')

    plt.fill_between(adjusted_iterations, smoothed_min_values, smoothed_max_values, alpha=0.5, color='lightgreen', label='Min-Max Range (Smoothed)')
    plt.plot(adjusted_iterations, smoothed_avg_values, label='Average (Smoothed)', color='green')

    plt.plot(adjusted_iterations, smoothed_min_values, label='Min (Smoothed)', color='red', linestyle='dashed')
    plt.plot(adjusted_iterations, smoothed_max_values, label='Max (Smoothed)', color='red', linestyle='dashed')

    plt.xlabel('Time Steps')
    plt.ylabel(ylabel)
    if log_scale:
        plt.yscale('log')
    elif symlog_scale:
        plt.yscale('symlog', linthresh=1e-5)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    path = 'plots'
    os.makedirs(path, exist_ok=True)

    filename = f'{path}/{name}.png'
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as {filename}")


def plot_training_results(
        tr_avg_undisc_returns,
        eval_avg_undisc_returns,
        eval_mean_traj_values,
        actor_losses,
        critic_losses,
):
    tr_returns = np.array(tr_avg_undisc_returns)
    eval_returns = np.array(eval_avg_undisc_returns)
    eval_trajec_values = np.array(eval_mean_traj_values)
    actor_losses = np.array(actor_losses)
    critic_losses = np.array(critic_losses)

    np.savez(
        'plots/plot_arrays.npz',
        tr_returns=tr_returns,
        eval_returns=eval_returns,
        eval_trajec_values=eval_trajec_values,
        actor_losses=actor_losses,
        critic_losses=critic_losses
    )

    tr_iterations = [1000 * i for i in range(1, len(tr_returns[0]) + 1)]
    eval_iterations = [20000 * i for i in range(1, len(eval_returns[0]) + 1)]

    aggregate_plot(
        tr_iterations,
        tr_returns,
        'Return',
        'Average Undiscounted Training Return',
        'tr_avg_undisc_return',
    )

    aggregate_plot(
        eval_iterations,
        eval_returns,
        'Return',
        'Average Undiscounted Evaluation Return',
        'eval_avg_undisc_return'
    )

    aggregate_plot(
        eval_iterations,
        eval_trajec_values,
        'Mean Value',
        'Mean Trajectory Value Function Over Evaluation',
        'eval_mean_traj_values'
    )

    aggregate_plot(
        tr_iterations,
        actor_losses,
        'Loss',
        'Actor Loss Over Training',
        'actor_losses',
        False,
    )

    aggregate_plot(
        tr_iterations,
        critic_losses,
        'Loss',
        'Critic Loss Over Training',
        'critic_losses',
        True
    )


def plot_values_over_trajectory(seed, values, n_iteration, save=True, display=False):
    time_steps = range(len(values))

    plt.figure(figsize=(10, 5))  # Adjust the figure size as needed

    plt.plot(time_steps, values, label='Value Function')

    plt.title('Value Function Over Trajectory')
    plt.xlabel('Time Steps')
    plt.ylabel('Value Estimate')
    plt.legend()
    plt.grid(True)

    path = f'plots/values_over_trajectory/seed_{seed}'
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
