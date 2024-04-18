import numpy as np
import torch
from torch import nn, optim
import gymnasium as gym
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
    print(K)
    for k in range(K):
        next_state, reward, term, trunc, _ = envs[k].step(actions[k])
        next_states.append(next_state)
        rewards.append(reward)
        print(envs[k].step(actions[k]))
        terminated = terminated or term
        truncated = truncated or trunc

    return torch.Tensor(np.array(next_states)), torch.Tensor(np.array(rewards)), terminated or truncated

def evaluate(actor, env, num_episodes=10):
    total_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        state = state[0] if isinstance(state, tuple) else state
        done = False
        total_reward = 0
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_probs = actor(state_tensor)
            action = torch.distributions.Categorical(action_probs).sample().item()
            next_state, reward, done, _ = env.step(action)
            next_state = next_state[0] if isinstance(next_state, tuple) else next_state
            total_reward += reward
            state = next_state
        total_rewards.append(total_reward)
    average_return = sum(total_rewards) / len(total_rewards)
    return average_return
def cart_pole_A2C(K: int = 1, n: int = 1, max_iter: int = 50000, eval_interval: int = 2000):

    envs = [gym.make('CartPole-v1') for _ in range(K)]
    nb_actions = envs[0].action_space.n
    nb_states = envs[0].observation_space.shape[0]
    actor = Actor(nb_states=nb_states, nb_actions=nb_actions)
    critic = Critic(nb_states=nb_states)
    gamma = 0.99
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-5)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
    done = False
    episode_rewards = []
    actor_losses = []
    critic_losses = []
    evaluations = []
    total_reward = 0
    t = 0
    states = torch.Tensor(np.array([envs[k].reset()[0] for k in range(K)]))
    all_success = False
    while not all_success and t < max_iter:
        print(t)
        print(done)
        # Sample actions for each environment
        policies = actor(states)
        actions = np.array([np.random.choice(np.arange(nb_actions), p=policy.detach().numpy()) for policy in policies])

        # Observe next states and rewards for each environment
        next_states, rewards, done = compute_next_states(envs, actions, K)
        if done:
            envs[0].reset()
            rewards = torch.zeros((K,0))
            done = False

        total_reward += rewards
        advantages = rewards + gamma * critic(next_states) - critic(states)

        # Update actor params
        actor_loss = torch.sum(advantages * torch.log(actor(states)[torch.arange(len(actions)), actions]))
        actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        actor_optimizer.step()
        critic_losses.append(actor_loss.item())

        # Update critic params
        critic_loss = torch.sum((rewards + gamma * critic(next_states) - critic(states)) ** 2)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        critic_losses.append(critic_loss.item())

        t += 1
        episode_rewards.append(rewards.mean().item())
        states = next_states

        if t%eval_interval == 0 or t == max_iter - 1:
            evaluation = evaluate(actor,envs)
            evaluations.append(t, evaluation)
            print(f"Evaluation at step {t}: Average Return = {evaluation}")
    return actor, critic, episode_rewards, actor_losses, critic_losses


actor, critic , episode_rewards, actor_losses, critic_losses = cart_pole_A2C(1, 1)
plot_training_results(episode_rewards,actor_losses,critic_losses)