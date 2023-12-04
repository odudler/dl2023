"Implementation of deep Q Agent, performing Q learning with neural networks"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from agent_interface import Agent
from utils import Memory, weights_init_

class DeepQAgent(Agent):
    def __init__(self, env, state_size: int = 42, action_size: int = 7, batch_size: int = 4):
        self.env = env

        self.state_size = state_size
        self.action_size = action_size
        self.memory = Memory(max_capacity=1000) # Replay memory
        self.gamma = 0.99 # Discount rate
        self.epsilon = 1.0 # Exploration rate (espilon-greedy)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.model = self._build_model()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.batch_size = batch_size

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )

        model.apply(weights_init_)

        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def load_model(self, loadpath):
        self.model.load_state_dict(torch.load(loadpath))
        self.model.eval()

    def save_model(self, savepath):
        torch.save(self.model.state_dict(), savepath)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = self.memory.sample(self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            # Predict next state
            target = reward
            if not done:
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.model(next_state_tensor.flatten())).item()
            state_tensor = torch.tensor(state, dtype=torch.float32)
            target_f = self.model(state_tensor.flatten())
            target_f[action] = target
            # Backpropagation
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state_tensor.flatten()))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def reset(self):
        self.memory.reset()

    def act(self, state):
        # Epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return self.env.random_valid_action()
        with torch.no_grad():
            q_values = self.model(torch.tensor(state, dtype=torch.float32).flatten())
            return np.argmax(q_values.numpy())