import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal, Categorical

class Actor(nn.Module):
    """
    Actor network for the A2C algorithm.

    Args:
        nb_states (int): Number of states.
        nb_actions (int): Number of actions.
        continuous (bool): If True, the action space is continuous.
    """
    def __init__(self, nb_states, nb_actions, continuous=False):
        super(Actor, self).__init__()
        self.continuous = continuous
        if continuous:
            self.network = nn.Sequential(
                nn.Linear(nb_states, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
        else:
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
    """
    Critic network for the A2C algorithm.

    Args:
        nb_states (int): Number of states.
    """
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
    """
    Actor-Critic model combining both actor and critic networks.

    Args:
        nb_states (int): Number of states.
        nb_actions (int): Number of actions.
        lr_actor (float): Learning rate for the actor.
        continuous (bool): If True, the action space is continuous.
    """
    def __init__(self, nb_states, nb_actions, lr_actor=1e-5, continuous=False):
        self.actor = Actor(nb_states, nb_actions, continuous)
        self.critic = Critic(nb_states)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
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
        """
        Samples an action based on the current policy.

        Args:
            state (torch.Tensor): Current state.

        Returns:
            torch.Tensor: Sampled action.
        """
        if self.continuous:
            mean = self.actor(state)
            std = self.actor.get_std()
            return Normal(mean, std).sample()
        else:
            policy = self.get_policy(state)
            return Categorical(policy).sample()

    def take_best_action(self, state):
        """
        Takes the best action based on the current policy.

        Args:
            state (torch.Tensor): Current state.

        Returns:
            np.ndarray or torch.Tensor: Best action.
        """
        if self.continuous:
            mean = self.actor(state)
            return torch.clamp(mean, -3, 3).detach().numpy()
        else:
            policy = self.get_policy(state).detach().numpy()
            return np.argmax(policy)

    def get_actions_prob(self, states, actions):
        """
        Gets the probability of the actions under the current policy.

        Args:
            states (torch.Tensor): Batch of states.
            actions (torch.Tensor): Batch of actions.

        Returns:
            torch.Tensor: Probabilities of the actions.
        """
        probas = torch.Tensor(states.shape[0])
        for k in range(states.shape[0]):
            probas[k] = self.get_policy(states[k])[actions[k]]
        return probas

    def update(self, targets, states, actions):
        """
        Updates the actor and critic networks.

        Args:
            targets (torch.Tensor): Target values.
            states (torch.Tensor): Batch of states.
            actions (torch.Tensor): Batch of actions.

        Returns:
            tuple: Actor loss and critic loss.
        """
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