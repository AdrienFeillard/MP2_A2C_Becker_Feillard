import numpy as np
import torch
import gymnasium as gym

import utils
from A2C import ActorCritic


def data_collection(
        state: np.array,
        nb_steps: int,
        env: gym.Env,
        actor_critic: ActorCritic,
        gamma: float,
        mask: bool
):
    discounted_return = 0.0
    step_state = state
    actions = []
    log_probs = []
    terminated = False
    truncated = False
    done = terminated or truncated
    undiscounted_return = 0.0

    step = 0
    while step < nb_steps and not done:
        # Compute action
        action, log_prob = actor_critic.sample_action(step_state)
        actions.append(action)
        log_probs.append(log_prob)
        next_state, reward, terminated, truncated, _ =  env.step(action.detach().numpy() if actor_critic.continuous else action.item())
        done = terminated or truncated

        discounted_reward = (gamma ** step) * float(reward)
        if mask:
            # 90% chance to zero out the reward
            discounted_reward = np.random.choice([0, discounted_reward], p=[0.9, 0.1])
        discounted_return += discounted_reward

        step_state = torch.Tensor(next_state)
        step += 1

        undiscounted_return += reward

    # TODO: for n steps always have batches of n
    with torch.no_grad():
        value = actor_critic.get_value(step_state)
    discounted_return += (gamma ** step) * (1 - terminated) * value

    return actions[0], step_state, discounted_return, undiscounted_return, done, log_probs[0]


def multistep_advantage_actor_critic(
        actor_critic: ActorCritic,
        gamma: float,
        n: int,
        max_iter: int,
        K: int = 1,
        mask: bool = False,
        env_name: str = 'CartPole-v1'
):
    debug_infos_interval = 1000
    evaluate_interval = 20000
    current_debug_threshold = debug_infos_interval
    current_eval_threshold = evaluate_interval

    envs = [gym.make(env_name) for _ in range(K)]
    states = torch.stack([torch.Tensor(envs[k].reset(seed=K * seed + k)[0]) for k in range(K)])

    it = 1

    episode_returns = [0 for _ in range(K)]
    episodes_returns = []
    episodes_returns_interval = []

    while it <= max_iter:
        actions = []
        next_states = []
        discounted_returns = torch.zeros(K)
        log_probs = []
        dones = [False for _ in range(K)]
        for k in range(K):
            action, next_state, discounted_return, undiscounted_return, done, log_prob = data_collection(
                states[k],
                n,
                envs[k],
                actor_critic,
                gamma,
                mask
            )
            actions.append(action)
            next_states.append(next_state)
            discounted_returns[k] = discounted_return
            log_probs.append(log_prob)
            episode_returns[k] += undiscounted_return
            dones[k] = done

        actor_loss, critic_loss = actor_critic.update(discounted_returns, states, actions, log_probs, n)

        done_returns = np.array(episode_returns)[np.array(dones)]
        if len(done_returns) > 0:
            mean_return = np.mean(done_returns)
            episodes_returns.append(mean_return)
            episodes_returns_interval.append(mean_return)

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
                nb_episodes=10,
                env_name=env_name
            )

            current_eval_threshold += evaluate_interval

        it += K * n
        states = torch.stack(next_states)

        for k in range(K):
            if dones[k]:
                episode_returns[k] = 0
                states[k] = torch.Tensor(envs[k].reset()[0])


def train_advantage_actor_critic(
        env_name: str = 'CartPole-v1',
        nb_actors: int = 1,
        nb_steps: int = 1,
        max_iter: int = 500000,
        gamma: float = 0.99,
        mask: bool = False
):
    env = gym.make(env_name)
    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]
    actor_critic = ActorCritic(nb_states, nb_actions, continuous=(env_name == 'InvertedPendulum-v4'))

    multistep_advantage_actor_critic(actor_critic, gamma, nb_steps, max_iter, nb_actors, mask, env_name)


def evaluate(
        actor_critic: ActorCritic,
        n_iteration,
        display_render=False,
        save_plot=True,
        display_plot=False,
        nb_episodes: int = 10,
        env_name: str = 'CartPole-v1'
):
    if display_render:
        render_mode = "human"
    else:
        render_mode = None
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
    nb_actors = int(input("Enter the number of parallel actors (default 1): ") or 1)
    nb_steps = int(input("Enter the number of steps for n-step returns (default 1): ") or 1)
    max_iter = int(input("Enter the maximum number of iterations (default 500000): ") or 500000)
    mask = input("Apply reward masking? (y/n, default 'n'): ").lower() == 'y'

    max_iterations = max_iter
    nb_seeds = 1
    K = nb_actors
    n = nb_steps
    gamma = 0.99

    tr_avg_undisc_returns = [[] for _ in range(nb_seeds)]
    eval_avg_undisc_returns = [[] for _ in range(nb_seeds)]
    eval_mean_traj_values = [[] for _ in range(nb_seeds)]
    actor_losses = [[] for _ in range(nb_seeds)]
    critic_losses = [[] for _ in range(nb_seeds)]

    for seed in range(nb_seeds):
        train_advantage_actor_critic(
            env_name=env,
            nb_actors=K,
            nb_steps=n,
            max_iter=max_iterations,
            gamma=gamma,
            mask=mask
        )

    utils.plot_training_results(
        tr_avg_undisc_returns,
        eval_avg_undisc_returns,
        eval_mean_traj_values,
        actor_losses,
        critic_losses,
    )
