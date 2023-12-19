# Base libraries
import numpy as np
import random
from typing import Union
import copy
import os

# ML libraries
import torch
import torch.nn as nn
import torch.optim as optim

# Local imports
from agents.agent_interface import Agent
from networks import DDQN
from utils import Memory
from env import Env

class DeepQAgent(Agent):
    """
    Implementation of the deep Q Agent, performing Q learning with neural networks.
    """
    def __init__(self, state_size: int = 42, action_size: int = 7, hidden_size: int = 128, hidden_layers: int = 4, batch_size: int = 10,
                 epsilon_max: float = 1.0, epsilon_min: float = 0.1, epsilon_decay: float = 0.999,
                 device: torch.device = torch.device("cpu"), options: Union[None, dict] = None):
        super(DeepQAgent, self).__init__(learning=True)

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size

        # Exploration rate (epsilon-greedy)
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_max
        self.epsilon_decay = epsilon_decay
        
        self.device = device
        
        # Replay memory
        self.memory = Memory(max_capacity=100000, device=self.device, min_capacity=100)

        # Parameters
        self.gamma = 1 # Discount rate, can be set to 1 for finite-horizon games
        
        self.num_optimizations = 0
        
        if type(options) == dict:
            self.options = options
            if type(self.options['weights_init']) == DeepQAgent: # Initialize with weights from passed model
                self.network = copy.deepcopy(self.options['weights_init'].network).to(self.device)
            else:
                raise ValueError(f'Cannot copy weigths to new model, invalid model type {type(self.options["weights_init"])}.')
        else: # No additional options passed, initialize new model
            self.network = DDQN(self.state_size, self.action_size, self.hidden_size, self.hidden_layers).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)

    def load_model(self, loadpath):
        self.network.load_state_dict(torch.load(loadpath, map_location=self.device))
        self.network.eval()
    
    def save_model(self, name: str = '', directory: str = './saved_models/'):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        if name == '': # If no name was given
            name = f'DQAgent_{self.num_optimizations}'
        torch.save(self.network.state_dict(), directory + name + '.pt')
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def optimize_model(self):
        if not self.memory.start_optimizing():
            return
        minibatch = self.memory.sample(self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            # Predict next state
            target = reward
            if not done:
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.network(next_state_tensor.flatten())).item()
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            target_f = self.network(state_tensor.flatten())
            target_f[action] = target
            # Backpropagation
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.network(state_tensor.flatten()))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.num_optimizations += 1

    def reset(self):
        self.memory.reset()
        self.num_optimizations = 0
        self.network = DDQN(self.state_size, self.action_size, self.hidden_size, self.hidden_layers).to(self.device)

    def act(self, env: Env):
        
        # Epsilon-greedy policy
        if random.random() > self.epsilon:
            state = env.get_state()
            state = torch.tensor(state, dtype=torch.float, device=self.device).reshape(1, -1)
            self.network.eval()
            with torch.no_grad():
                q_values = self.network(state)
            self.network.train()
            action = torch.argmax(q_values, dim=1)
            action = int(action)
        else:
            action = env.random_valid_action()
        return action