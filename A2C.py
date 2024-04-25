import numpy as np
import torch
from torch import nn, optim


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


class ActionCritic:
    def __init__(self, nb_states, nb_actions):
        self.actor = Actor(nb_states, nb_actions)
        self.critic = Critic(nb_states)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.nb_actions = nb_actions

    def get_policy(self, state):
        return self.actor(state)

    def get_value(self, state):
        return self.critic(state)

    def sample_action(self, state):
        policy = self.get_policy(state).detach().numpy()
        return np.random.choice(np.arange(self.nb_actions), p=policy)

    def take_best_action(self, state):
        policy = self.get_policy(state).detach().numpy()
        return np.argmax(policy)

    def update(self, discounted_returns, state, action):
        advantage = discounted_returns - self.get_value(state)

        # Update actor params
        actor_loss = -advantage * torch.log(self.get_policy(state)[action])
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        # Update critic params
        critic_loss = (discounted_returns - self.get_value(state)) ** 2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss.detach(), critic_loss.detach()
