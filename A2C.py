import numpy as np
import torch
from torch import nn, optim
import gymnasium as gym

import utils
from utils import *


class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(nb_states, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, nb_actions),
            nn.Softmax(dim=-1),
        )

    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    def __init__(self, nb_states):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(nb_states, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.network(state)


def compute_next_states(envs, actions, K):
    next_states, rewards, terminated, truncated = [], [], False, False
    for k in range(K):
        next_state, reward, term, trunc, _ = envs[k].step(actions[k])
        next_states.append(next_state)
        rewards.append(reward)
        terminated = terminated or term
        truncated = truncated or trunc

    return torch.Tensor(np.array(next_states)), torch.Tensor(np.array(rewards)), terminated, truncated


def evaluate(K, actor, critic, num_episodes=10):
    envs = [gym.make('CartPole-v1') for _ in range(K)]

    episode_returns = []
    plot_states = []
    plot_values = []
    for e in range(num_episodes):
        states = torch.Tensor(np.array([envs[k].reset()[0] for k in range(K)]))
        done = False
        returns = torch.zeros((K,))
        while not done:
            policies = actor(states)
            actions = np.array(
                [np.random.choice(np.arange(envs[0].action_space.n), p=policy.detach().numpy()) for policy in policies]
            )
            next_states, rewards, terminated, truncated = compute_next_states(envs, actions, K)
            done = truncated or terminated
            returns += rewards

            if e == num_episodes - 1:
                plot_states.append(states[0].detach())
                plot_values.append(critic(states[0]).detach().item())

            states = next_states
        episode_returns.append(returns.mean().detach().item())

    utils.plot_critic_values(np.array(plot_states), plot_values)

    return np.mean(episode_returns)


def data_collection(K, nb_steps, gamma, actor, envs, states):
    returns = torch.zeros((K,))
    tmp_states = states
    steps_actions = []
    done = False

    for n in range(nb_steps):
        # Sample actions for each environment
        policies = actor(tmp_states)
        steps_actions.append(
            np.array([np.random.choice(np.arange(envs[0].action_space.n), p=policy.detach().numpy()) for policy in
                      policies]))

        # Observe next states and rewards for each environment
        next_states, rewards, terminated, truncated = compute_next_states(envs, steps_actions[n], K)

        returns += gamma ** n * rewards
        tmp_states = next_states

        if terminated or truncated:
            done = True
            break

    return returns, steps_actions[0], tmp_states, done


def update_actor_critic(actor, actor_optimizer, actor_loss, critic, critic_optimizer, critic_loss):
    # Update actor params
    actor_optimizer.zero_grad()
    actor_loss.backward(retain_graph=True)
    actor_optimizer.step()

    # Update critic params
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    return actor, critic


def cart_pole_A2C(K: int = 1, nb_steps: int = 1, max_iter: int = 500000, eval_interval: int = 20000,
                  debug_infos_interval: int = 1000, gamma: float = 0.99):
    envs = [gym.make('CartPole-v1') for _ in range(K)]

    # Create actor and critic
    nb_actions = envs[0].action_space.n
    nb_states = envs[0].observation_space.shape[0]
    actor = Actor(nb_states=nb_states, nb_actions=nb_actions)
    critic = Critic(nb_states=nb_states)

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-5)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    episode_returns = []
    actor_losses = []
    critic_losses = []
    evaluations = []
    t = 0
    states = torch.Tensor(np.array([envs[k].reset()[0] for k in range(K)]))

    print_debug_values = False

    while t < max_iter:
        # Collect the returns, actions and next states in step t
        returns, actions, next_states, done = data_collection(K, nb_steps, gamma, actor, envs, states)
        returns += gamma ** nb_steps * torch.flatten(critic(next_states))
        episode_returns.append(returns.mean().detach().item())
        advantages = returns - critic(states)

        # Update actor and critic parameters
        actor_loss = torch.sum(advantages * torch.log(actor(states)[torch.arange(len(actions)), actions]))
        actor_losses.append(actor_loss.item())
        critic_loss = torch.sum((returns - critic(states)) ** 2)
        critic_losses.append(critic_loss.item())
        actor, critic = update_actor_critic(actor, actor_optimizer, actor_loss, critic, critic_optimizer, critic_loss)

        if t % debug_infos_interval == 0:
            print(f"Actor loss = {actor_loss}")
            print(f"Critic loss = {critic_loss}")

        t += 1
        # episode_rewards.append(rewards.mean().item())
        states = next_states

        if t % eval_interval == 0 or t == max_iter:
            evaluation = evaluate(K, actor, critic)
            # evaluations.append(t, evaluation)
            print(f"Evaluation at step {t}: Average Return = {evaluation}")

        if done:
            print(f"Step {t}: Average Return = {np.mean(episode_returns)}")
            states = torch.Tensor(np.array([envs[k].reset()[0] for k in range(K)]))

    return actor, critic


actor, critic = cart_pole_A2C()
