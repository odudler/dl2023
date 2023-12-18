"Implementation of deep Q Agent, performing Q learning with neural networks"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from typing import Union
import os

from .agent_interface import Agent
from networks import DDQN
from utils import Memory, weights_init_
from env import Env

class DeepQAgent(Agent):
    def __init__(self, state_size: int = 42, action_size: int = 7, hidden_size: int = 128, hidden_layers: int = 4, batch_size: int = 10, epsilon_max: float = 1.0, epsilon_min: float = 0.1,
                epsilon_decay: float = 0.999, device: str = "cpu", options: Union[None, dict] = None):
        super(DeepQAgent, self).__init__(learning=True)

        self.device = device

        self.state_size = state_size
        self.action_size = action_size
        self.memory = Memory(max_capacity=10000) # Replay memory
        self.gamma = 1 # Discount rate, can be set to 1 for finite-horizon games
        self.epsilon = epsilon_max # Exploration rate (espilon-greedy)
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        if type(options) == dict:
            self.options = options

            if type(self.options['weights_init']) == DeepQAgent: # Initialize with weights from passed model
                self.model = copy.deepcopy(self.options['weights_init'].model).to(self.device)
            else:
                raise ValueError(f'Cannot copy weigths to new model, invalid model type {type(self.options["weights_init"])}.')
        else: # No additional options passed, initialize new model
            self.model = DDQN(state_size, action_size, hidden_size, hidden_layers).apply(weights_init_).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.batch_size = batch_size

        self.num_optimizations = 0
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def load_model(self, loadpath):
        self.model.load_state_dict(torch.load(loadpath))
        self.model.eval()
    
    def save_model(self, name: str = '', directory: str = './saved_models/'):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        if name == '': # If no name was given
            name = f'DQAgent_{self.num_optimizations}'
        torch.save(self.model.state_dict(), directory + name)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = self.memory.sample(self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            # Predict next state
            target = reward
            if not done:
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.model(next_state_tensor.flatten())).item()
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            target_f = self.model(state_tensor.flatten())
            target_f[action] = target
            # Backpropagation
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state_tensor.flatten()))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.num_optimizations += 1

    def reset(self):
        self.memory.reset()

    def act(self, env: Env):
        # Epsilon-greedy policy
        if random.random() > self.epsilon:
            state = env.get_state()
            with torch.no_grad():
                q_values = self.model(torch.tensor(state, dtype=torch.float32, device=self.device).flatten())
                return torch.argmax(q_values).cpu().numpy()
        return env.random_valid_action()