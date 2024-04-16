import numpy as np
import torch
from torch import nn, optim
import gymnasium as gym


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

    return torch.Tensor(np.array(next_states)), torch.Tensor(np.array(rewards)), terminated or truncated


def cart_pole_A2C(K: int = 1, n: int = 1, max_iter: int = 50000):
    envs = [gym.make('CartPole-v1') for _ in range(K)]

    # Create actor and critic
    nb_actions = envs[0].action_space.n
    nb_states = envs[0].observation_space.shape[0]
    actor = Actor(nb_states=nb_states, nb_actions=nb_actions)
    critic = Critic(nb_states=nb_states)

    gamma = 0.99

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-5)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    done = False

    t = 0
    states = torch.Tensor(np.array([envs[k].reset()[0] for k in range(K)]))

    while not done and t < max_iter:
        # Sample actions for each environment
        policies = actor(states)
        actions = np.array([np.random.choice(np.arange(nb_actions), p=policy.detach().numpy()) for policy in policies])

        # Observe next states and rewards for each environment
        next_states, rewards, done = compute_next_states(envs, actions, K)

        total_rewards = rewards + gamma * critic(next_states)
        advantages = total_rewards - critic(states)

        # Update actor params
        actor_loss = torch.sum(advantages * torch.log(actor(states)[torch.arange(len(actions)), actions]))
        actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        actor_optimizer.step()

        # Update critic params
        critic_loss = torch.sum((total_rewards - critic(states)) ** 2)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        t += 1
        states = next_states

    return actor, critic


actor, critic = cart_pole_A2C(4, 1)
