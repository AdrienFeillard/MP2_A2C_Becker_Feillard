import os

import matplotlib.pyplot as plt
import numpy as np


def aggregate_plot(iterations, values, ylabel, title, name, log_scale=False, symlog_scale=False):
    assert not log_scale or not symlog_scale, "Cannot use both log and symlog scale at the same time"

    plt.figure(figsize=(10, 5))  # Adjust the figure size as needed

    min_values = np.min(values, axis=0)
    max_values = np.max(values, axis=0)
    avg_values = np.mean(values, axis=0)

    plt.fill_between(iterations, min_values, max_values, alpha=0.5, label='Min-Max Range')
    plt.plot(iterations, avg_values, label='Average')

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
        True
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
