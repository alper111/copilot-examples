"""Proximal policy optimization with PyTorch"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# pylint: disable=invalid-name, too-many-locals, too-many-arguments, too-many-instance-attributes

class Policy(nn.Module):
    """Policy network"""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """Forward pass"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

class PPO:
    """Proximal policy optimization agent"""
    def __init__(self, state_dim, action_dim, hidden_dim, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = Policy(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = Policy(state_dim, action_dim, hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        """Select action"""
        state = torch.from_numpy(state).float().to(device)
        probs = self.policy_old(state)
        m = Categorical(probs)
        action = m.sample()
        self.policy_old.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def update(self, memory):
        """Update policy"""
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(self.policy_old.saved_log_probs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def evaluate(self, state, action):
        """Evaluate policy"""
        probs = self.policy(state)
        m = Categorical(probs)

        # Finding the entropy
        dist_entropy = m.entropy()

        # Finding action log probabilities
        action_logprobs = m.log_prob(action)

        # Finding state values
        state_values = self.policy(state)

        return action_logprobs, state_values, dist_entropy

def main():
    """Main"""
    # creating environment
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # creating agent
    agent = PPO(state_dim, action_dim, args.hidden_dim, args.lr, args.betas, args.gamma, args.K_epochs, args.eps_clip)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    for i_episode in range(1, args.max_episodes+1):
        state = env.reset()
        for t in range(args.max_timesteps):
            time_step += 1

            # Running policy_old:
            action = agent.select_action(state)

            # Saving reward and is_terminal:
            state, reward, done, _ = env.step(action)
            agent.policy_old.rewards.append(reward)
            agent.policy_old.is_terminals.append(done)

            # update if its time
            if time_step % args.update_timestep == 0:
                agent.update(agent.policy_old)
                agent.policy_old.clear_memory()
                time_step = 0

            running_reward += reward
            if done:
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward > (args.solved_reward * i_episode):
            print("########## Solved! ##########")
            torch.save(agent.policy.state_dict(), './PPO_{}.pth'.format(args.env_name))
            break

        # save every 100 episodes
        if i_episode % 100 == 0:
            torch.save(agent.policy.state_dict(), './PPO_{}.pth'.format(args.env_name))

        # logging
        if i_episode % args.log_interval == 0:
            avg_length = int(avg_length/args.log_interval)
            running_reward = int((running_reward/args.log_interval))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

if __name__ == '__main__':
    main()


# I am using the following command to run the code:

# python3 ppo.py --env-name CartPole-v0 --max-episodes 10000 --max-timesteps 200 --update-timestep 2000 --log-interval 10 --solved-reward 195 --hidden-dim 64 --lr 0.002 --betas 0.9 0.999 --gamma 0.99 --K-epochs 4 --eps-clip 0.2