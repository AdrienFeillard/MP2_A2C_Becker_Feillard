from typing import List

import numpy as np
import torch
import gymnasium as gym

import utils
from A2C import ActorCritic


def data_collection(
        states: torch.Tensor,
        envs: List[gym.Env],
        actor_critic: ActorCritic,
        K: int,
        n: int,
        gamma: float,
        mask: bool
):
    actions = [[None for _ in range(n)] for _ in range(K)]
    next_states = [[torch.empty([]) for _ in range(n)] for _ in range(K)]
    terminated_arr = [[False for _ in range(n)] for _ in range(K)]
    truncated_arr = [[False for _ in range(n)] for _ in range(K)]
    dones = [[False for _ in range(n)] for _ in range(K)]

    rewards = torch.zeros((K, n))
    logging_rewards = torch.zeros((K, n))

    for k in range(K):
        tmp_state = states[k]
        for i in range(n):
            # Compute action
            action = actor_critic.sample_action(tmp_state)
            next_state, reward, terminated, truncated, _ = envs[k].step(action.item())
            if terminated or truncated:
                next_state = envs[k].reset()[0]
            actions[k][i] = action
            next_states[k][i] = torch.Tensor(next_state)
            terminated_arr[k][i] = terminated
            truncated_arr[k][i] = truncated
            dones[k][i] = terminated or truncated

            rewards[k, i] = float(reward)
            logging_rewards[k, i] = float(reward)

            if mask:
                # 90% chance to zero out the reward
                rewards[k, i] = np.random.choice([0, rewards[k, i]], p=[0.9, 0.1])

            tmp_state = next_states[k][i]

    targets = torch.zeros((K, n))
    for k in range(K):
        for i in range(n):
            target = 0
            j_end = n-1
            for j in range(i, n):
                target += (gamma ** (j - i)) * rewards[k, j]
                if terminated_arr[k][j] or truncated_arr[k][j]:
                    j_end = j
                    break
            with torch.no_grad():
                value = actor_critic.get_value(next_states[k][j_end])
            target += (gamma ** (j_end - i + 1)) * (1 - terminated_arr[k][j_end]) * value.squeeze()
            targets[k, i] = target

    actions = torch.stack([torch.stack(actions[k]) for k in range(K)])
    next_states = torch.stack([torch.stack(next_states[k]) for k in range(K)])

    return actions, next_states, targets, logging_rewards, dones


def multistep_advantage_actor_critic(
        actor_critic: ActorCritic,
        gamma: float,
        n: int,
        max_iter: int,
        K: int = 1,
        mask: bool = False
):
    debug_infos_interval = 1000
    evaluate_interval = 20000
    current_debug_threshold = debug_infos_interval
    current_eval_threshold = evaluate_interval

    envs = [gym.make('CartPole-v1') for _ in range(K)]
    states = torch.stack([torch.Tensor(envs[k].reset(seed=K * seed + k)[0]) for k in range(K)])

    it = 1

    episode_returns = [0 for _ in range(K)]
    episodes_returns = []
    episodes_returns_interval = []

    while it <= max_iter:
        actions, next_states, targets, rewards, dones = data_collection(
            states,
            envs,
            actor_critic,
            K,
            n,
            gamma,
            mask
        )
        states = torch.cat([states[:, None, :], next_states], dim=1)

        actor_loss, critic_loss = actor_critic.update(targets, states[:, :-1], actions)

        for i in range(rewards.shape[1]):
            for k in range(rewards.shape[0]):
                episode_returns[k] += rewards[k, i]
            done_returns = np.array(episode_returns)[np.array(dones)[:, i]]
            if len(done_returns) > 0:
                mean_return = np.mean(done_returns)
                episodes_returns.append(mean_return)
                episodes_returns_interval.append(mean_return)
                for k in range(rewards.shape[0]):
                    if dones[k][i]:
                        episode_returns[k] = 0

        if it >= current_debug_threshold:
            # Keep latest available return
            tr_avg_undisc_returns[seed].append(episodes_returns[-1])

            average_reward = np.mean(episodes_returns_interval)

            # Reset for the next 1000 steps
            episodes_returns_interval = []
            print(
                f"At step {it}:\n"
                f"\tAverage Return of episodes in last {debug_infos_interval} steps = {average_reward}"
            )

            actor_losses[seed].append(actor_loss.item())
            critic_losses[seed].append(critic_loss.item())
            print(f"\tActor loss = {actor_loss}")
            print(f"\tCritic loss = {critic_loss}\n")

            current_debug_threshold += debug_infos_interval

        if it >= current_eval_threshold:
            evaluate(
                actor_critic,
                it,
                display_render=False,
                save_plot=True,
                display_plot=False,
                nb_episodes=10
            )

            current_eval_threshold += evaluate_interval

        it += K * n
        states = states[:, -1]


def train_advantage_actor_critic(
        nb_actors: int = 1,
        nb_steps: int = 1,
        max_iter: int = 500000,
        gamma: int = 0.99,
        mask: bool = False
):
    env = gym.make('CartPole-v1')
    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.n
    actor_critic = ActorCritic(nb_states, nb_actions)

    multistep_advantage_actor_critic(actor_critic, gamma, nb_steps, max_iter, nb_actors, mask=mask)


def evaluate(
        actor_critic: ActorCritic,
        n_iteration,
        display_render=False,
        save_plot=True,
        display_plot=False,
        nb_episodes: int = 10
):
    if display_render:
        render_mode = "human"
    else:
        render_mode = None
    env = gym.make('CartPole-v1', render_mode=render_mode)

    episode_values = []
    episode_returns = []
    plot_states = []
    plot_values = []

    for e in range(nb_episodes):
        state, _ = env.reset()
        state = torch.Tensor(state)
        done = False
        undiscounted_return = 0

        while not done:
            action = actor_critic.take_best_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            undiscounted_return += reward
            done = truncated or terminated

            if render_mode is not None:
                env.render()

            value = actor_critic.get_value(state).item()
            episode_values.append(value)
            if e == nb_episodes - 1:
                plot_states.append(state.detach())
                plot_values.append(actor_critic.get_value(state).detach().item())

            state = torch.Tensor(next_state)
        episode_returns.append(undiscounted_return)

    mean_return = np.mean(episode_returns)
    eval_avg_undisc_returns[seed].append(mean_return.item())
    print(f"\nMean undiscounted return for evaluation at step {n_iteration} = {mean_return}")

    # After collecting all values
    # Calculate and log the mean value function over the trajectory
    mean_value = np.mean(episode_values)
    eval_mean_traj_values[seed].append(mean_value.item())

    utils.plot_values_over_trajectory(seed, plot_values, n_iteration, save=save_plot, display=display_plot)


if __name__ == '__main__':
    max_iterations = 500000
    nb_seeds = 3
    K = 6
    n = 1
    mask = True

    tr_avg_undisc_returns = [[] for _ in range(nb_seeds)]
    eval_avg_undisc_returns = [[] for _ in range(nb_seeds)]
    eval_mean_traj_values = [[] for _ in range(nb_seeds)]
    actor_losses = [[] for _ in range(nb_seeds)]
    critic_losses = [[] for _ in range(nb_seeds)]

    for seed in range(nb_seeds):
        train_advantage_actor_critic(K, n, max_iter=max_iterations, mask=mask)

    utils.plot_training_results(
        tr_avg_undisc_returns,
        eval_avg_undisc_returns,
        eval_mean_traj_values,
        actor_losses,
        critic_losses,
    )
