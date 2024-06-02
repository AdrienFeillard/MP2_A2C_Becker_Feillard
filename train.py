import numpy as np
import torch
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

import utils
from A2C import ActorCritic


def data_collection(
        states: torch.Tensor,
        envs: AsyncVectorEnv,
        actor_critic: ActorCritic,
        K: int,
        n: int,
        gamma: float,
        mask: bool
):
    actions = []
    next_states = torch.zeros((K, n, *states.shape[1:]), dtype=torch.float32)
    terminated_arr = torch.zeros((K, n), dtype=torch.bool)
    truncated_arr = torch.zeros((K, n), dtype=torch.bool)
    dones = torch.zeros((K, n), dtype=torch.bool)

    rewards = torch.zeros((K, n), dtype=torch.float32)
    logging_rewards = torch.zeros((K, n), dtype=torch.float32)

    for i in range(n):
        # Compute actions for all environments
        action_batch = [actor_critic.sample_action(states[k]) for k in range(K)]
        actions.append(action_batch)
        action_batch = [torch.clamp(action, -3, 3).detach().numpy() if actor_critic.continuous else action.item() for action in action_batch]

        next_state, reward, terminated, truncated, _ = envs.step(action_batch)

        next_states[:, i] = torch.tensor(next_state, dtype=torch.float32)
        terminated_arr[:, i] = torch.tensor(terminated, dtype=torch.bool)
        truncated_arr[:, i] = torch.tensor(truncated, dtype=torch.bool)
        dones[:, i] = terminated_arr[:, i] | truncated_arr[:, i]

        rewards[:, i] = torch.tensor(reward, dtype=torch.float32)
        logging_rewards[:, i] = torch.tensor(reward, dtype=torch.float32)

        if mask:
            # 90% chance to zero out the reward
            rewards[:, i] = torch.tensor([np.random.choice([0, reward], p=[0.9, 0.1]) for reward in rewards[:, i]], dtype=torch.float32)

        states = next_states[:, i]

    targets = torch.zeros((K, n), dtype=torch.float32)
    for k in range(K):
        for i in range(n):
            target = 0
            j_end = n - 1
            for j in range(i, n):
                target += (gamma ** (j - i)) * rewards[k, j]
                if terminated_arr[k, j] or truncated_arr[k, j]:
                    j_end = j
                    break
            with torch.no_grad():
                value = actor_critic.get_value(next_states[k, j_end])
            target += (gamma ** (j_end - i + 1)) * torch.logical_not(terminated_arr[k, j_end]).float() * value.squeeze()
            targets[k, i] = target

    actions = torch.stack([torch.stack(actions[k]) for k in range(K)])
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

    env_name = 'InvertedPendulum-v4' if actor_critic.continuous else 'CartPole-v1'
    envs = AsyncVectorEnv([lambda: gym.make(env_name) for _ in range(K)])
    observations, _ = envs.reset(seed=42)
    states = torch.tensor(observations, dtype=torch.float32)

    total_steps = 0
    episode_returns = [0 for _ in range(K)]
    episodes_returns = []
    episodes_returns_interval = []

    while total_steps <= max_iter:
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

        if total_steps >= current_debug_threshold:
            # Keep latest available return
            tr_avg_undisc_returns[seed].append(episodes_returns[-1])

            average_reward = np.mean(episodes_returns_interval)

            # Reset for the next 1000 steps
            episodes_returns_interval = []
            print(
                f"At step {total_steps}:\n"
                f"\tAverage Return of episodes in last {debug_infos_interval} steps = {average_reward}"
            )

            actor_losses[seed].append(actor_loss.item())
            critic_losses[seed].append(critic_loss.item())
            print(f"\tActor loss = {actor_loss}")
            print(f"\tCritic loss = {critic_loss}\n")

            current_debug_threshold += debug_infos_interval

        if total_steps >= current_eval_threshold:
            evaluate(
                actor_critic,
                total_steps,
                display_render=False,
                save_plot=True,
                display_plot=False,
                nb_episodes=10
            )

            current_eval_threshold += evaluate_interval

        total_steps += K * n
        states = states[:, -1]


def train_advantage_actor_critic(
        env_name: str,
        nb_actors: int = 1,
        nb_steps: int = 1,
        max_iter: int = 500000,
        gamma: int = 0.99,
        lr_actor: float = 1e-5,
        mask: bool = False
):
    env = gym.make(env_name)
    nb_states = env.observation_space.shape[0]
    continuous = env_name == 'InvertedPendulum-v4'
    nb_actions = 1 if continuous else env.action_space.n
    actor_critic = ActorCritic(nb_states, nb_actions, lr_actor, continuous)

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
    env_name = 'InvertedPendulum-v4' if actor_critic.continuous else 'CartPole-v1'
    env = gym.make(env_name, render_mode=render_mode)

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
    env_choice = input("Enter the environment (1 for 'CartPole-v1', 2 for 'InvertedPendulum-v4'): ")
    if env_choice == '1':
        env = 'CartPole-v1'
    elif env_choice == '2':
        env = 'InvertedPendulum-v4'
    else:
        raise ValueError("Invalid input! Please enter 1 or 2.")
    K = int(input("Enter the number of parallel actors (default 1): ") or 1)
    n = int(input("Enter the number of steps for n-step returns (default 1): ") or 1)
    lr_actor = float(input("Enter the learning rate for the actor (default 1e-5): ") or 1e-5)
    max_iterations = int(input("Enter the maximum number of iterations (default 500000): ") or 500000)
    mask = (input("Apply reward masking? (y/n, default 'y'): ") or 'y') == 'y'

    nb_seeds = 3

    tr_avg_undisc_returns = [[] for _ in range(nb_seeds)]
    eval_avg_undisc_returns = [[] for _ in range(nb_seeds)]
    eval_mean_traj_values = [[] for _ in range(nb_seeds)]
    actor_losses = [[] for _ in range(nb_seeds)]
    critic_losses = [[] for _ in range(nb_seeds)]

    for seed in range(nb_seeds):
        train_advantage_actor_critic(env, K, n, lr_actor=lr_actor, max_iter=max_iterations, mask=mask)

    utils.plot_training_results(
        tr_avg_undisc_returns,
        eval_avg_undisc_returns,
        eval_mean_traj_values,
        actor_losses,
        critic_losses,
    )