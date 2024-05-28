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


class ActorCritic:
    def __init__(self, *args):
        if len(args) == 2:
            nb_states, nb_actions = args[0], args[1]
            self.actor = Actor(nb_states, nb_actions)
            self.critic = Critic(nb_states)

            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

            self.nb_actions = nb_actions

        elif len(args) == 3:
            nb_actions, actor_copy, critic_copy = args[0], args[1], args[2]
            self.actor = actor_copy
            self.critic = critic_copy

            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

            self.nb_actions = nb_actions

        else:
            raise ValueError("Unexpected arguments for ActorCritic constructor.")

    def get_policy(self, state):
        return self.actor(state)

    def get_value(self, state):
        return self.critic(state)

    def sample_action(self, state):
        policy = self.get_policy(state)
        dist = torch.distributions.Categorical(policy)
        return dist.sample()

    def take_best_action(self, state):
        policy = self.get_policy(state).detach().numpy()
        return np.argmax(policy)

    def update(self, discounted_returns, state, action, n, K):
        value = self.get_value(state)
        with torch.no_grad():
            advantage = discounted_returns - value

        # Update actor params
        actor_loss = (-advantage * torch.log(self.get_policy(state)[action])) / (n * K)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Use semi-gradient instead of full gradient
        # critic_loss = ((discounted_returns - self.get_value(state)) ** 2) / (n * K)

        # Update critic params
        critic_loss = ((discounted_returns - value) ** 2) / (n * K)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss.detach(), critic_loss.detach()

    def copy(self):
        actor_copy = self.actor
        actor_copy.load_state_dict(self.actor.state_dict())

        critic_copy = self.critic
        critic_copy.load_state_dict(self.critic.state_dict())

        return ActorCritic(self.nb_actions, actor_copy, critic_copy)

    def share_memory(self):
        self.actor.share_memory()
        self.critic.share_memory()

