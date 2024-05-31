import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal, Categorical


class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, continuous=False):
        super(Actor, self).__init__()
        self.continuous = continuous
        self.network = nn.Sequential(
            nn.Linear(nb_states, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, nb_actions),
            nn.Softmax(dim=-1),
        )
        self.log_std = nn.Parameter(torch.zeros(nb_actions)) if continuous else None

    def forward(self, state):
        return self.network(state)

    def get_std(self):
        return torch.exp(self.log_std)


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
    def __init__(self, nb_states, nb_actions, continuous=False):
        self.actor = Actor(nb_states, nb_actions, continuous)
        self.critic = Critic(nb_states)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.nb_actions = nb_actions
        self.continuous = continuous

    def get_policy(self, state):
        if self.continuous:
            mean = self.actor(state)
            std = self.actor.get_std()
            return lambda action: torch.exp(Normal(mean, std).log_prob(action))

        return self.actor(state)

    def get_value(self, state):
        return self.critic(state)

    def sample_action(self, state):
        if self.continuous:
            mean = self.actor(state)
            std = self.actor.get_std()
            return Normal(mean, std).sample()
        else:
            policy = self.get_policy(state)
            return Categorical(policy).sample()

    def take_best_action(self, state):
        if self.continuous:
            mean = self.actor(state)
            return torch.clamp(mean, -3, 3).detach().numpy()
        else:
            policy = self.get_policy(state).detach().numpy()
            return np.argmax(policy)

    def get_actions_prob(self, states, actions):
        probas = torch.Tensor(states.shape[0])
        for k in range(states.shape[0]):
            probas[k] = self.get_policy(states[k])[actions[k]]
        return probas

    def update(self, targets, states, actions):
        targets = targets.reshape(-1)
        states = states.reshape(-1, states.shape[-1])
        actions = actions.reshape(-1)

        values = self.get_value(states).squeeze()

        with torch.no_grad():
            advantages = targets - values

        if self.continuous:
            mean = self.actor(states)
            std = self.actor.get_std()
            log_probs = Normal(mean, std).log_prob(actions)
        else:
            log_probs = torch.log(self.get_actions_prob(states, actions))

        # Update actor params
        actor_loss = torch.mean(-advantages * log_probs)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update critic params
        critic_loss = torch.mean((targets - values) ** 2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss.detach(), critic_loss.detach()
