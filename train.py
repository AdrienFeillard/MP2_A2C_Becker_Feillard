import numpy as np
import torch
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter

import utils
from A2C import ActorCritic


def data_collection(state: np.array, nb_steps: int, env: gym.Env, actor_critic: ActorCritic, gamma: float):
    discounted_returns = 0.0
    step_state = state
    actions = []
    terminated = False
    truncated = False
    done = terminated or truncated
    total_reward = 0.0

    step = 0
    while step < nb_steps and not done:
        # Compute action
        action = actor_critic.sample_action(step_state)
        actions.append(action)
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        discounted_returns += (gamma ** step) * float(reward)
        step_state = torch.Tensor(next_state)
        step += 1

        total_reward += reward

    # TODO: for n steps always have batches of n
    with torch.no_grad():
        value = actor_critic.get_value(step_state)
    discounted_returns += (gamma ** step) * (1 - terminated) * value

    return actions[0], step_state, discounted_returns, total_reward, done


def multistep_advantage_actor_critic_episode(
        env: gym.Env,
        actor_critic: ActorCritic,
        iteration: int,
        gamma: float,
        state: torch.Tensor,
        episode_rewards: list,
        nb_steps: int = 1,
        max_iter: int = 500000,
        k: int = 1,
):
    """
    Run an episode of multistep A2C on the given environment.
    """

    total_reward = 0
    done = False

    debug_infos_interval = 1000
    evaluate_interval = 20000

    while iteration <= max_iter and not done:
        action, next_state, discounted_returns, rewards, done = data_collection(
            state,
            nb_steps,
            env,
            actor_critic,
            gamma
        )
        total_reward += rewards

        actor_loss, critic_loss = actor_critic.update(discounted_returns, state, action, nb_steps, k)

        # writer.add_scalar('Loss/Actor', actor_loss.item(), iteration)
        # writer.add_scalar('Loss/Critic', critic_loss.item(), iteration)
        actor_losses[seed].append(actor_loss)
        critic_losses[seed].append(critic_loss)

        if iteration % debug_infos_interval == 0:
            average_reward = np.mean(episode_rewards)
            # writer.add_scalar('Training/Average Undiscounted Return', average_reward, iteration)
            tr_avg_undisc_returns[seed].append(average_reward)
            print(f"\nAt step {iteration}: \n\tAverage Reward of last episodes = {average_reward}")
            print(f"\tActor loss = {actor_loss}")
            print(f"\tCritic loss = {critic_loss}")
            # Reset for the next 1000 steps
            episode_rewards = []

        if iteration % evaluate_interval == 0:
            evaluate(
                actor_critic,
                k,
                nb_steps,
                iteration,
                display_render=False,
                save_plot=True,
                display_plot=False,
                nb_episodes=10
            )

        iteration += 1
        state = next_state

    return total_reward, episode_rewards, iteration


def multistep_advantage_actor_critic(
        actor_critic: ActorCritic,
        gamma: float,
        nb_steps: int,
        max_iter: int,
        k: int = 1
):
    episodes_rewards = []

    env = gym.make('CartPole-v1')
    state, _ = env.reset(seed=seed)

    it = 1
    while it <= max_iter:
        # Run one episode of A2C
        total_reward, episodes_rewards, it = multistep_advantage_actor_critic_episode(
            env=env,
            actor_critic=actor_critic,
            iteration=it,
            gamma=gamma,
            state=torch.Tensor(state),
            episode_rewards=episodes_rewards,
            nb_steps=nb_steps,
            max_iter=max_iter,
            k=k,
        )
        episodes_rewards.append(total_reward)
        state, _ = env.reset()


def train_advantage_actor_critic(nb_actors: int = 1, nb_steps: int = 1, max_iter: int = 500000, gamma: int = 0.99):
    env = gym.make('CartPole-v1')
    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.n
    actor_critic = ActorCritic(nb_states, nb_actions)

    multistep_advantage_actor_critic(actor_critic, gamma, nb_steps, max_iter, nb_actors)


def evaluate(
        actor_critic: ActorCritic,
        actor: int,
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
    # writer.add_scalar('Evaluation/Mean_Undiscounted_Return', mean_return, n_iteration)
    eval_avg_undisc_returns[seed].append(mean_return)
    print(f"\nMean undiscounted return for evaluation at step {n_iteration} = {mean_return}")
    # utils.plot_values_over_trajectory(plot_values, K, n_steps, n_iteration, save_plot, display_plot)

    # After collecting all values
    # Calculate and log the mean value function over the trajectory
    mean_value = np.mean(episode_values)
    # writer.add_scalar('Evaluation/Mean_Value_Function', mean_value, n_iteration)
    eval_mean_traj_values[seed].append(mean_value)

    utils.plot_values_over_trajectory(seed, plot_values, actor, n_iteration, save=save_plot, display=display_plot)
    """
    for timestep, value in enumerate(last_episode_values):
        writer.add_scalar(f'Evaluation/Trajectories/Evaluation_{n_iteration // 20000}_Value_Function_Last_Episode', value, timestep)
    """


if __name__ == '__main__':
    nb_seeds = 3
    empty = [[] for _ in range(nb_seeds)]
    tr_avg_undisc_returns = empty.copy()
    eval_avg_undisc_returns = empty.copy()
    eval_mean_traj_values = empty.copy()
    actor_losses = empty.copy()
    critic_losses = empty.copy()

    for seed in range(nb_seeds):
        # writer = SummaryWriter(f'runs/Seed_{seed}')
        train_advantage_actor_critic(1, 1, max_iter=500000)


