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
from utils import Memory, soft_update
from env import Env

class DeepQAgent(Agent):
    """
    Implementation of the deep Q Agent, performing Q learning with neural networks.
    """
    def __init__(self, env: Env, state_size: int = 42, action_size: int = 7, hidden_size: int = 128, hidden_layers: int = 4, batch_size: int = 10,
                 epsilon_max: float = 1.0, epsilon_min: float = 0.1, epsilon_decay: float = 0.999,
                 device: torch.device = torch.device("cpu"), options: Union[None, dict] = None):
        super(DeepQAgent, self).__init__(learning=True)
        
        self.env = env
        
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
        self.memory = Memory(max_capacity=100000, min_capacity=100, device=self.device)

        # Parameters
        self.gamma = 1 # Discount rate, can be set to 1 for finite-horizon games
        self.tau = 1e-3 # Soft update param
        
        self.num_optimizations = 0
        
        if type(options) == dict:
            self.options = options
            if type(self.options['weights_init']) == DeepQAgent: # Initialize with weights from passed model
                self.network = copy.deepcopy(self.options['weights_init'].network).eval().to(self.device)
                self.target_net = copy.deepcopy(self.options['weights_init'].target_net).eval().to(self.device)
            else:
                raise ValueError(f'Cannot copy weigths to new model, invalid model type {type(self.options["weights_init"])}.')
        else: # No additional options passed, initialize new model
            self.network = DDQN(self.state_size, self.action_size, self.hidden_size, self.hidden_layers).eval().to(self.device)
            self.target_net = DDQN(self.state_size, self.action_size, self.hidden_size, self.hidden_layers).eval().to(self.device)
            self.target_net.load_state_dict(self.network.state_dict())
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)

    def load_model(self, loadpath: str):
        self.network.load_state_dict(torch.load(loadpath, map_location=self.device))
        self.network.eval()
        self.target_net.load_state_dict(torch.load(loadpath, map_location=self.device))
        self.target_net.eval()
    
    def save_model(self, name: str = '', directory: str = './saved_models/'):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        if name == '': # If no name was given
            name = f'DQAgent_{self.num_optimizations}'
        torch.save(self.network.state_dict(), directory + name + '.pt')
        
    def remember(self, state: list, action: list, reward: list, next_state: list, done: list):
        self.memory.push(state, action, reward, next_state, done)

    def optimize_model(self):
        if not self.memory.start_optimizing():
            return
        
        self.network.train()
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, split_transitions=True)
        
        move_validity = next_states[:, 0, :] == 0 # For full columns, q-values should be -inf (-1e7) since this move will be invalid (See below next_q_values)
        states = states.reshape(self.batch_size, -1).to(self.device)
        actions = actions.to(torch.int64).reshape(self.batch_size, -1).to(self.device)
        rewards = rewards.reshape(self.batch_size, -1).to(self.device)
        next_states = next_states.reshape(self.batch_size, -1).to(self.device)
        dones[dones >= 0] = 1 # Game finished (finished == 0, 1, 2)
        dones[dones < 0] = 0 # Game did not finish (finished == -1)
        dones = dones.reshape(self.batch_size, -1).to(self.device)
        
        q_values = self.network(states).gather(dim=1, index=actions) # Get Q-Values for the actions 
        next_q_values = self.target_net(next_states).detach() # Detach since no gradient calc needed
        next_q_values = torch.where(move_validity, next_q_values, -1e7).max(dim=1, keepdim=True)[0] # Get max Q-Values for the next_states.
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        soft_update(self.network, self.target_net, self.tau)
        
        self.num_optimizations += 1

    def reset(self):
        self.memory.reset()
        self.num_optimizations = 0
        self.network = DDQN(self.state_size, self.action_size, self.hidden_size, self.hidden_layers).eval().to(self.device)
        self.target_net = DDQN(self.state_size, self.action_size, self.hidden_size, self.hidden_layers).eval().to(self.device)
        self.target_net.load_state_dict(self.network.state_dict())

    def act(self, state: torch.Tensor, **kwargs):
        # Ensure state is on correct device and in correct form
        state = state.to(self.device).float()

        # Parse input for determinstic keyword argument
        if getattr(kwargs, 'deterministic', None) != None:
            deterministic = kwargs['deterministic']
        else:
            deterministic = False

        # Epsilon-greedy policy
        if deterministic or random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.network(state.flatten())
                return torch.argmax(q_values).cpu().numpy()
        return self.env.random_valid_action()