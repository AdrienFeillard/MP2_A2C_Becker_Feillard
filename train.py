from typing import List

import numpy as np
import torch
import gymnasium as gym
import torch.multiprocessing as mp

from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Manager

import utils
from A2C import ActorCritic



def data_collection(state: np.array, nb_steps: int, env: gym.Env, actor_critic: ActorCritic, gamma: float):
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


def multistep_advantage_actor_critic_episode(
        env: gym.Env,
        actor_critic: ActorCritic,
        iteration: mp.Value,
        gamma: float,
        lock: mp.Lock,
        nb_steps: int = 1,
        max_iter: int = 500000,
        k: int = 1,

) -> float:
    """
    Run an episode of multistep A2C on the given environment.
    :return: tuple containing the (total) reward for the episode
    """

    total_reward = 0
    done = False

    state, _ = env.reset()
    state = torch.Tensor(state)
    episode_rewards = []
    training_rewards = []
    training_actor_losses = []
    training_critic_losses = []
    step_count = 0
    debug_infos_interval = 1000
    evaluate_interval = 20000

    while iteration.value <= max_iter and not done:
        action, next_state, discounted_returns, rewards, done = data_collection(
            state,
            nb_steps,
            env,
            actor_critic.copy(),
            gamma
        )
        total_reward += rewards
        step_count += nb_steps
        lock.acquire()
        episode_rewards.append(total_reward)
        writer.add_scalar('Training/Total Reward', total_reward, iteration.value)

        try:
            actor_loss, critic_loss = actor_critic.update(discounted_returns, state, action)

            training_actor_losses.append(actor_loss)
            training_critic_losses.append(critic_loss)

            writer.add_scalar('Loss/Actor', actor_loss.item(), iteration.value)
            writer.add_scalar('Loss/Critic', critic_loss.item(), iteration.value)

            if iteration.value % debug_infos_interval == 0:
                print(f"\nIteration {iteration.value}: \n\tActor loss = {actor_loss} \n\tCritic loss = {critic_loss}")
                #average_reward = total_reward / step_count
                average_reward = np.mean(episode_rewards)
                print(f"\nAt step {iteration.value}: Average Reward of last episodes = {average_reward}")
                # Reset for the next 1000 steps
                episode_rewards = []
                step_count = 0
            if iteration.value % evaluate_interval == 0:
                avg_return = evaluate(actor_critic, k, nb_steps, iteration.value, display_render=False,
                                      save_plot=True,
                                      display_plot=False, nb_episodes=10)
                print(f"\nEvaluation at iteration {iteration.value}: \n\tAverage Return = {avg_return}")

            iteration.value += 1
        finally:
            lock.release()

        state = next_state

    #print(f"\nEnd of episode at iteration {iteration.value - 1}: \n\tEpisode reward = {total_reward}")

    return total_reward, training_rewards, training_actor_losses, training_critic_losses, episode_rewards


def multistep_advantage_actor_critic(
        actor_critic: ActorCritic,
        it: mp.Value,
        gamma: float,
        nb_steps: int,
        max_iter: int,
        lock: mp.Lock,
        k: int = 1,
):
    episodes_rewards = []
    training_rewards = []
    training_actor_losses = []
    training_critic_losses = []
    env = gym.make('CartPole-v1')

    while it.value <= max_iter:
        # Run one episode of A2C
        total_reward, training_rewards, training_actor_losses, training_critic_losses, episode_rewards = (
            multistep_advantage_actor_critic_episode(
                env=env,
                actor_critic=actor_critic,
                iteration=it,
                gamma=gamma,
                nb_steps=nb_steps,
                max_iter=max_iter,
                lock=lock,
                k=k,
            ))

        episodes_rewards.append(total_reward)
        training_rewards.append(training_rewards)
        training_actor_losses.append(training_actor_losses)
        training_critic_losses.append(training_critic_losses)
    utils.plot_training_results(episodes_rewards, training_rewards, training_actor_losses, training_critic_losses,
                                show_plot=False, save_plot=True)


def train_advantage_actor_critic(nb_actors: int = 1, nb_steps: int = 1, max_iter: int = 500000, gamma: int = 0.99):
    env = gym.make('CartPole-v1')
    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.n

    actor_critic = ActorCritic(nb_states, nb_actions)
    it = mp.Value('i', 1)

    manager = Manager()
    lock = manager.Lock()
    actor_critic.share_memory()
    processes = []
    for _ in range(nb_actors):
        process = mp.Process(
            target=multistep_advantage_actor_critic,
            args=(actor_critic, it, gamma, nb_steps, max_iter, lock, nb_actors)
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


def evaluate(actor_critic: ActorCritic, K, n_steps, n_iteration, display_render=False, save_plot=True,
             display_plot=False,
             nb_episodes: int = 10):
    if display_render:
        render_mode = 'human'
    else:
        render_mode = None
    env = gym.make('CartPole-v1', render_mode=render_mode)

    episode_values =[]
    episode_returns = []
    plot_states = []
    plot_values = []
    value_during_episode = []
    print(nb_episodes)
    for e in range(nb_episodes):
        state, _ = env.reset()
        state = torch.Tensor(state)
        done = False
        undiscounted_return = 0
        print(e)
        while not done:
            action = actor_critic.take_best_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            undiscounted_return += reward
            done = truncated or terminated
            env.render()
            writer.add_scalar(f'Evaluation/Reward_{e}', undiscounted_return, n_iteration)
            writer.add_scalar(f'Evaluation/Value_{e}', actor_critic.get_value(state).item(), n_iteration)
            value = actor_critic.get_value(state).item()
            value_during_episode.append(value)
            if e == nb_episodes - 1:
                #print("last episode",e)
                plot_states.append(state.detach())
                plot_values.append(actor_critic.get_value(state).detach().item())
                episode_values.append(actor_critic.get_value(state).item())
                """
                for timestep, value in enumerate(value_during_episode):
                    #print(timestep,value)
                    writer.add_scalar('Evaluation/Value_Function', value, timestep)
                """
            for timestep, value in enumerate(episode_values):
                writer.add_scalar('Evaluation/Value_Function_Last_Episode', value, timestep)

            state = torch.Tensor(next_state)

        episode_returns.append(undiscounted_return)
    print(f"\nAverage Reward of 10 episodes at evaluation = {np.mean(episode_returns)}")
    utils.plot_critic_values(np.array(plot_states), plot_values, K, n_steps, n_iteration, save_plot, display_plot)

    # After collecting all values
    # Calculate and log the mean value function over the trajectory
    mean_value = np.mean(value_during_episode)
    writer.add_scalar('Evaluation/Mean_Value_Function', mean_value, n_iteration)
    return np.mean(episode_returns)


writer = SummaryWriter('runs/advantage_actor_critic_experiment')

if __name__ == '__main__':
    """
    for seed in range(3):
    """

    train_advantage_actor_critic(1, 1, max_iter=500000)
