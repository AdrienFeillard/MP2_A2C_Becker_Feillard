import numpy as np
import torch
import gymnasium as gym
import torch.multiprocessing as mp

from torch.utils.tensorboard import SummaryWriter

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
        barrier: mp.Barrier,
        episode_rewards: list,
        nb_steps: int = 1,
        max_iter: int = 500000,
        k: int = 1
):
    """
    Run an episode of multistep A2C on the given environment.
    """

    total_reward = 0
    done = False

    state, _ = env.reset()
    state = torch.Tensor(state)

    training_rewards = []
    training_actor_losses = []
    training_critic_losses = []

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

        barrier.wait()
        actor_loss, critic_loss = actor_critic.update(discounted_returns, state, action, nb_steps, k)
        barrier.wait()

        training_actor_losses.append(actor_loss)
        training_critic_losses.append(critic_loss)

        writer.add_scalar('Loss/Actor', actor_loss.item(), iteration.value)
        writer.add_scalar('Loss/Critic', critic_loss.item(), iteration.value)

        if iteration.value % debug_infos_interval == 0:
            average_reward = np.mean(episode_rewards)
            writer.add_scalar('Training/Average Undiscounted Return', average_reward, iteration.value)
            print(f"\nAt step {iteration.value}: \n\tAverage Reward of last episodes = {average_reward}")
            print(f"\tActor loss = {actor_loss}")
            print(f"\tCritic loss = {critic_loss}")
            # Reset for the next 1000 steps
            episode_rewards = []

        if iteration.value % evaluate_interval == 0:
            avg_return = evaluate(
                actor_critic,
                k,
                nb_steps,
                iteration.value,
                display_render=False,
                save_plot=True,
                display_plot=False,
                nb_episodes=10
            )
            print(f"\nEvaluation at iteration {iteration.value}: \n\tAverage Return = {avg_return}")

        # TODO: Check for n-steps
        iteration.value += 1

        state = next_state

    return total_reward, training_rewards, training_actor_losses, training_critic_losses, episode_rewards


def multistep_advantage_actor_critic(
        actor_critic: ActorCritic,
        it: mp.Value,
        gamma: float,
        nb_steps: int,
        max_iter: int,
        barrier: mp.Barrier,
        k: int = 1
):
    episodes_rewards = []
    training_rewards = []
    training_actor_losses = []
    training_critic_losses = []

    env = gym.make('CartPole-v1')

    while it.value <= max_iter:
        # Run one episode of A2C
        total_reward, training_rewards, training_actor_losses, training_critic_losses, ep_rewards = (
            multistep_advantage_actor_critic_episode(
                env=env,
                actor_critic=actor_critic,
                iteration=it,
                gamma=gamma,
                episode_rewards=episodes_rewards,
                nb_steps=nb_steps,
                max_iter=max_iter,
                barrier=barrier,
                k=k
            )
        )
        episodes_rewards = ep_rewards

        episodes_rewards.append(total_reward)
        training_rewards.append(training_rewards)
        training_actor_losses.append(training_actor_losses)
        training_critic_losses.append(training_critic_losses)

    utils.plot_training_results(
        episodes_rewards,
        training_rewards,
        training_actor_losses,
        training_critic_losses,
        show_plot=False,
        save_plot=True
    )


def train_advantage_actor_critic(nb_actors: int = 1, nb_steps: int = 1, max_iter: int = 500000, gamma: int = 0.99):
    env = gym.make('CartPole-v1')
    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.n
    env.reset(seed=5)
    actor_critic = ActorCritic(nb_states, nb_actions)
    it = mp.Value('i', 1)

    barrier = mp.Barrier(nb_actors)
    actor_critic.share_memory()
    processes = []
    for _ in range(nb_actors):
        process = mp.Process(
            target=multistep_advantage_actor_critic,
            args=(actor_critic, it, gamma, nb_steps, max_iter, barrier, nb_actors)
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


def evaluate(
        actor_critic: ActorCritic,
        K,
        n_steps,
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
    value_during_episode = []

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
            # writer.add_scalar(f'Evaluation/Reward_{e}', undiscounted_return, n_iteration)
            # writer.add_scalar(f'Evaluation/Value_{e}', actor_critic.get_value(state).item(), n_iteration)
            value = actor_critic.get_value(state).item()
            value_during_episode.append(value)
            if e == nb_episodes - 1:
                plot_states.append(state.detach())
                plot_values.append(actor_critic.get_value(state).detach().item())
                episode_values.append(actor_critic.get_value(state).item())
                """
                for timestep, value in enumerate(value_during_episode):
                    #print(timestep,value)
                    writer.add_scalar('Evaluation/Value_Function', value, timestep)
                """


            state = torch.Tensor(next_state)
        for timestep, value in enumerate(episode_values):
            print(timestep)
            print("value: ", value)
            writer.add_scalar('Evaluation/Value_Function_Last_Episode', value, timestep)
        episode_returns.append(undiscounted_return)

    mean_return = np.mean(episode_returns)
    writer.add_scalar('Evaluation/Average Undiscounted Return', mean_return, n_iteration)
    print(f"\nAverage undiscounted return for evaluation at step {n_iteration}:\n\t{mean_return}")
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

    train_advantage_actor_critic(1, 1, max_iter=50000)
