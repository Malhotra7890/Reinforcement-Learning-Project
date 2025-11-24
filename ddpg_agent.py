import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

STATE_DIM = 8
ACTION_DIM = 2
HIDDEN = 256

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, ACTION_DIM),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1)
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.net(x)

class ReplayBuffer:
    def __init__(self, max_size=200000):
        self.buffer = deque(maxlen=max_size)

    def add(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r), np.array(s2), np.array(d))

class DDPGAgent:
    def __init__(self, gamma=0.99, tau=0.005, actor_lr=1e-4, critic_lr=1e-3):
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor()
        self.critic = Critic()

        self.actor_target = Actor()
        self.critic_target = Critic()

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay = ReplayBuffer()
        self.batch_size = 256

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).cpu().detach().numpy()[0]
        action += np.random.normal(0, noise, size=ACTION_DIM)
        return np.clip(action, -1, 1)

    def train(self):
        if len(self.replay.buffer) < self.batch_size:
            return

        s, a, r, s2, d = self.replay.sample(self.batch_size)
        s = torch.FloatTensor(s)
        a = torch.FloatTensor(a)
        r = torch.FloatTensor(r).unsqueeze(1)
        s2 = torch.FloatTensor(s2)
        d = torch.FloatTensor(d).unsqueeze(1)

        # Critic loss
        with torch.no_grad():
            target_actions = self.actor_target(s2)
            target_q = self.critic_target(s2, target_actions)
            y = r + self.gamma * (1 - d) * target_q

        q = self.critic(s, a)
        critic_loss = nn.MSELoss()(q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actor_loss = -self.critic(s, self.actor(s)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
