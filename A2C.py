import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal


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
        if continuous:
            self.log_std = nn.Parameter(torch.zeros(nb_actions))

    def forward(self, state):
        mean = self.network(state)
        if self.continuous:
            std = torch.exp(self.log_std)
            return mean, std
        return mean


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
        return self.actor(state)

    def get_value(self, state):
        return self.critic(state)

    def sample_action(self, state):
        if self.continuous:
            mean, std = self.get_policy(state)
            dist = Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, -3, 3)  # Clip actions to the range [-3, 3]
            return action, dist.log_prob(action).sum()
        else:
            policy = self.get_policy(state)
            dist = torch.distributions.Categorical(policy)
            return dist.sample(), dist.log_prob(dist.sample())

    def take_best_action(self, state):
        if self.continuous:
            mean, _ = self.get_policy(state)
            return torch.clamp(mean, -3, 3).detach().numpy()
        else:
            policy = self.get_policy(state).detach().numpy()
            return np.argmax(policy)

    def get_actions_prob(self, states, actions):
        probas = torch.Tensor(states.shape[0])
        for k in range(states.shape[0]):
            probas[k] = self.get_policy(states[k])[actions[k]]
        return probas

    def update(self, targets, states, actions, log_probs):
        targets = targets.reshape(-1)
        states = states.reshape(-1, states.shape[-1])
        actions = actions.reshape(-1)

        values = self.get_value(states).squeeze()

        with torch.no_grad():
            advantages = targets - values

        # Update actor params
        if isinstance(log_probs,list):
            log_probs = torch.stack(log_probs)
        if self.continuous:
            actor_loss = (-advantages * log_probs).mean()
        else:
            actor_loss = (-advantages * torch.log(self.get_actions_prob(states, actions))).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update critic params
        critic_loss = torch.mean((targets - values) ** 2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss.detach(), critic_loss.detach()

    def copy(self):
        actor_copy = self.actor
        actor_copy.load_state_dict(self.actor.state_dict())

        critic_copy = self.critic
        critic_copy.load_state_dict(self.critic.state_dict())

        continuous_copy = self.continuous

        return ActorCritic(self.nb_actions, actor_copy, critic_copy, continuous_copy)

    def share_memory(self):
        self.actor.share_memory()
        self.critic.share_memory()
