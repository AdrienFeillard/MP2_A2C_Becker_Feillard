from typing import Tuple

import numpy as np
import torch
import gymnasium as gym

import utils
from A2C import ActionCritic


def multistep_advantage_actor_critic(
        env: gym.Env,
        actor_critic: ActionCritic,
        iteration: int,
        gamma: float,
        nb_steps: int = 1,
        max_iter: int = 500000
) -> Tuple[float, int]:
    """
    Run an episode of multistep A2C on the given environment.
    :return: tuple containing the (total) reward for the episode, and the number of steps
    """

    total_reward = 0
    done = False

    state, _ = env.reset()
    state = torch.Tensor(state)

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

        actor_loss, critic_loss = actor_critic.update(discounted_returns, state, action)

        if iteration % debug_infos_interval == 0:
            print(f"\nIteration {iteration}: \n\tActor loss = {actor_loss} \n\tCritic loss = {critic_loss}")

        if iteration % evaluate_interval == 0:
            avg_return = evaluate(actor_critic)
            print(f"\nEvaluation at iteration {iteration}: \n\tAverage Return = {avg_return}")

        iteration += 1
        state = next_state

    print(f"\nEnd of episode at iteration {iteration-1}: \n\tEpisode reward = {total_reward}")

    return total_reward, iteration


def data_collection(state: np.array, nb_steps: int, env: gym.Env, actor_critic: ActionCritic, gamma: float):
    discounted_returns = 0.0
    step_state = state
    actions = []
    terminated = False
    truncated = False
    total_reward = 0.0

    step = 0
    while step < nb_steps and not truncated:
        # Compute action
        action = actor_critic.sample_action(step_state)
        actions.append(action)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        discounted_returns += gamma ** step * float(reward)
        step_state = torch.Tensor(next_state)
        step += 1

    discounted_returns += gamma ** step * (1 - terminated) * actor_critic.get_value(step_state)

    return actions[0], step_state, discounted_returns, total_reward, terminated or truncated


def train_advantage_actor_critic(nb_actors: int = 1, nb_steps: int = 1, max_iter: int = 500000, gamma: int = 0.99):
    env = gym.make('CartPole-v1')

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.n
    actor_critic = ActionCritic(nb_states, nb_actions)

    it = 1
    while it <= max_iter:
        # Run one episode of A2C
        total_reward, it = multistep_advantage_actor_critic(
            env=env,
            actor_critic=actor_critic,
            iteration=it,
            gamma=gamma,
            nb_steps=nb_steps,
            max_iter=max_iter
        )


def evaluate(actor_critic: ActionCritic):
    env = gym.make('CartPole-v1')

    episode_returns = []
    plot_states = []
    plot_values = []

    nb_episodes = 10
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

            if e == nb_episodes - 1:
                plot_states.append(state.detach())
                plot_values.append(actor_critic.get_value(state).detach().item())

            state = torch.Tensor(next_state)
        episode_returns.append(undiscounted_return)

    utils.plot_critic_values(np.array(plot_states), plot_values)

    return np.mean(episode_returns)


train_advantage_actor_critic()
