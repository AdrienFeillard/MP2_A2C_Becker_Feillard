import os
import numpy as np
import matplotlib.pyplot as plt

def exponential_moving_average(data, alpha=0.3):
    """
    Computes the exponential moving average of the provided data.

    Args:
        data (np.ndarray): Input data.
        alpha (float): Smoothing factor.

    Returns:
        np.ndarray: Smoothed data.
    """
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t - 1]
    return ema

def aggregate_plots(tr_iterations, eval_iterations, array, agents, ylabel, title, name, log_scale=False, smoothing=False, plot_eval=False):
    """
    Plots the aggregate results with optional smoothing and evaluation data.

    Args:
        tr_iterations (list): Training iteration steps.
        eval_iterations (list): Evaluation iteration steps.
        array (list): Values to plot.
        agents (list): List of agent identifiers.
        ylabel (str): Y-axis label.
        title (str): Plot title.
        name (str): Filename for saving the plot.
        log_scale (bool): If True, use logarithmic scale for y-axis.
        smoothing (bool): If True, apply smoothing.
        plot_eval (bool): If True, include evaluation data in the plot.
    """
    plt.figure(figsize=(10, 5))  # Adjust the figure size as needed

    if smoothing:
        # Apply smoothing
        min_values = np.min(array[0], axis=0)
        max_values = np.max(array[0], axis=0)
        avg_values = np.mean(array[0], axis=0)

        alpha = 0.1  # Smoothing factor

        smoothed_min_values = exponential_moving_average(min_values, alpha)
        smoothed_max_values = exponential_moving_average(max_values, alpha)
        smoothed_avg_values = exponential_moving_average(avg_values, alpha)

        adjusted_iterations = tr_iterations[:len(smoothed_avg_values)]

        plt.plot(tr_iterations, avg_values, label='Average (Raw)', color='steelblue')

        plt.fill_between(adjusted_iterations, smoothed_min_values, smoothed_max_values, alpha=0.5, color='lightsalmon', label='Min-Max Range (Smoothed)')
        plt.plot(adjusted_iterations, smoothed_avg_values, label='Average (Smoothed)', color='red')

        if plot_eval:
            min_values = np.min(array[1], axis=0)
            max_values = np.max(array[1], axis=0)
            avg_values = np.mean(array[1], axis=0)
            plt.fill_between(eval_iterations, min_values, max_values, alpha=0.5, label=f'Agent {agents[1]} Min-Max Range', color='thistle')
            plt.plot(eval_iterations, avg_values, label=f'Agent {agents[1]} Average', color='violet', linestyle='-.')
    else:
        for i in range(len(agents)):
            min_values = np.min(array[i], axis=0)
            max_values = np.max(array[i], axis=0)
            avg_values = np.mean(array[i], axis=0)

            plt.fill_between(eval_iterations, min_values, max_values, alpha=0.5, label=f'Agent {agents[i]} Min-Max Range')
            plt.plot(eval_iterations, avg_values, label=f'Agent {agents[i]} Average')

    plt.xlabel('Time Steps')
    plt.ylabel(ylabel)
    if log_scale:
        plt.yscale('log')
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

    """
    Loads data from plot arrays and generates plots for training and evaluation results.
    """
    plots_arrays = []
    for i in range(1, 8):
        path = f'plots/agent{i}/plot_arrays.npz' if i != 7 else f'plots/agent1_stoch/plot_arrays.npz'
        plots_arrays.append(np.load(path))

    tr_iterations = [1000 * i for i in range(1, len(plots_arrays[0]['tr_returns'][0]))]
    eval_iterations = [20000 * i for i in range(1, len(plots_arrays[0]['eval_returns'][0]))]

    tr_returns_arrays = [plots_arrays[i]['tr_returns'] for i in range(7)]
    eval_returns_arrays = [plots_arrays[i]['eval_returns'] for i in range(7)]
    eval_trajec_values_arrays = [plots_arrays[i]['eval_trajec_values'] for i in range(7)]
    actor_arrays = [plots_arrays[i]['actor_losses'] for i in range(7)]
    critic_arrays = [plots_arrays[i]['critic_losses'] for i in range(7)]


    aggregate_plots(
        tr_iterations,
        eval_iterations,
        [eval_trajec_values_arrays[-1][:, :-1], eval_trajec_values_arrays[1], eval_trajec_values_arrays[2], eval_trajec_values_arrays[3]],
        ['1_stoch', '2', '3', '4'],
        'Critic value',
        None,
        'eval_trajec_values_1_2_3_4',
        log_scale=False
    )

    aggregate_plots(
        tr_iterations,
        eval_iterations,
        [eval_trajec_values_arrays[4][:, :-1], eval_trajec_values_arrays[5]],
        ['5', '6'],
        'Critic value',
        None,
        'eval_trajec_values_5_6',
        log_scale=False
    )

    aggregate_plots(
        tr_iterations,
        eval_iterations,
        [eval_trajec_values_arrays[0][:,:24]],
        ['1'],
        'Critic value',
        None,
        'eval_trajec_values_1',
        log_scale=False
    )

    aggregate_plots(tr_iterations,
                    eval_iterations,
                    [actor_arrays[0][:,:499]],
                    ['1'],
                    'Actor loss',
                    None,
                    'actor_loss_1',
                    log_scale=False,
                    smoothing=True)

    aggregate_plots(tr_iterations,
                    eval_iterations,
                    [actor_arrays[-1][:,:499]],
                    ['1 stoch'],
                    'Actor loss',
                    None,
                    'actor_loss_1_stoch',
                    log_scale=False,
                    smoothing=True)

    aggregate_plots(tr_iterations,
                    eval_iterations,
                    [actor_arrays[1][:,:499]],
                    ['2'],
                    'Actor loss',
                    None,
                    'actor_loss_2',
                    log_scale=False,
                    smoothing=True)
    aggregate_plots(tr_iterations,
                    eval_iterations,
                    [actor_arrays[2][:,:499]],
                    ['3'],
                    'Actor loss',
                    None,
                    'actor_loss_3',
                    log_scale=False,
                    smoothing=True)

    aggregate_plots(tr_iterations,
                    eval_iterations,
                    [actor_arrays[3][:,:499]],
                    ['4'],
                    'Actor loss',
                    None,
                    'actor_loss_4',
                    log_scale=False,
                    smoothing=True)

    aggregate_plots(tr_iterations,
                    eval_iterations,
                    [actor_arrays[4][:,:499]],
                    ['5'],
                    'Actor loss',
                    None,
                    'actor_loss_5',
                    log_scale=False,
                    smoothing=True)

    aggregate_plots(tr_iterations,
                    eval_iterations,
                    [actor_arrays[5][:,:499]],
                    ['6'],
                    'Actor loss',
                    None,
                    'actor_loss_6',
                    log_scale=False,
                    smoothing=True)

    aggregate_plots(tr_iterations,
                    eval_iterations,
                    [critic_arrays[0][:,:499]],
                    ['1'],
                    'Critic loss',
                    None,
                    'critic_loss_1',
                    log_scale=True,
                    smoothing=True)

    aggregate_plots(tr_iterations,
                    eval_iterations,
                    [critic_arrays[1][:,:499]],
                    ['2'],
                    'Critic loss',
                    None,
                    'critic_loss_2',
                    log_scale=True,
                    smoothing=True)

    aggregate_plots(tr_iterations,
                    eval_iterations,
                    [critic_arrays[2][:,:499]],
                    ['3'],
                    'Critic loss',
                    None,
                    'critic_loss_3',
                    log_scale=True,
                    smoothing=True)

    aggregate_plots(tr_iterations,
                    eval_iterations,
                    [critic_arrays[3][:,:499]],
                    ['4'],
                    'Critic loss',
                    None,
                    'critic_loss_4',
                    log_scale=True,
                    smoothing=True)

    aggregate_plots(tr_iterations,
                    eval_iterations,
                    [critic_arrays[4][:,:499]],
                    ['5'],
                    'Critic loss',
                    None,
                    'critic_loss_5',
                    log_scale=True,
                    smoothing=True)

    aggregate_plots(tr_iterations,
                    eval_iterations,
                    [critic_arrays[5][:,:499]],
                    ['6'],
                    'Critic loss',
                    None,
                    'critic_loss_6',
                    log_scale=True,
                    smoothing=True)

    aggregate_plots(tr_iterations,
                    eval_iterations,
                    [critic_arrays[6][:,:499]],
                    ['1_stoch'],
                    'Critic loss',
                    None,
                    'critic_loss_1_stoch',
                    log_scale=True,
                    smoothing=True)

    ############
    # Plot training and evaluation returns for each agent
    for i in range(7):
        aggregate_plots(
            tr_iterations,
            eval_iterations,
            [tr_returns_arrays[i][:, :499], eval_returns_arrays[i][:, :24]],
            [f'{i+1} training', f'{i+1} evaluation'],
            'Undiscounted return',
            f'tr_return_{i+1}',
            log_scale=False,
            smoothing=True,
            plot_eval=True
        )

def plot_values_over_trajectory(seed, values, n_iteration, name, save=True, display=False):
    """
    Plots the value function over a trajectory.

    Args:
        seed (int): Random seed.
        values (list): Value estimates.
        n_iteration (int): Current iteration number.
        save (bool): If True, save the plot.
        display (bool): If True, display the plot.
    """
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
        filename = f'{path}/{name}/iter_{n_iteration}.png'
        plt.savefig(filename)
        print(f"Plot saved as {filename}")

    # Display the plot if the display flag is True
    if display:
        plt.show()
    else:
        plt.close()