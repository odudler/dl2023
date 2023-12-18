"""Implementation of CQL DQN agent. 
For more information see https://github.com/BY571/CQL/tree/main/CQL-DQN."""

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
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

# Local imports
from agents.agent_interface import Agent
from networks import DDQN
from utils import Memory, weights_init_, soft_update
from env import Env

class CQLAgent(Agent):
    def __init__(self, state_size: int = 42, action_size: int = 7, hidden_size: int = 64, hidden_layers: int = 3, batch_size: int = 4, 
                 epsilon_max: float = 1.0, epsilon_min: float = 0.1, epsilon_decay: float = 0.999,
                 device: torch.device = torch.device("cpu"), options: Union[None, dict] = None):
        super(CQLAgent, self).__init__(learning=True)
        
        self.device = device

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.memory = Memory(max_capacity=10000, device=self.device) # Replay memory
        self.batch_size = batch_size

        self.epsilon_max = epsilon_max
        self.epsilon = epsilon_max # Exploration rate (epsilon-greedy)
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.gamma = 1 # Discount rate
        self.tau = 1e-3 # Soft update param

        self.num_optimizations = 0

        if type(options) == dict:
            self.options = options

            if type(self.options['weights_init']) == CQLAgent: # Initialize with weights from passed model
                self.network = copy.deepcopy(self.options['weights_init'].network).eval().to(self.device)
                self.target_net = copy.deepcopy(self.options['weights_init'].target_net).eval().to(self.device)
            else:
                raise ValueError(f'Cannot copy weigths to new model, invalid model type {type(self.options["weights_init"])}.')
        else: # No additional options passed, initialize new model
            self.network = DDQN(state_size, action_size, hidden_size, hidden_layers).apply(weights_init_).to(self.device)
            self.target_net = DDQN(state_size, action_size, hidden_size, hidden_layers).apply(weights_init_).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)

    def load_model(self, loadpath):
        self.network.load_state_dict(torch.load(loadpath, map_location=torch.device(self.device)))
        self.network.eval()

        self.target_net.load_state_dict(torch.load(loadpath, map_location=torch.device(self.device)))
        self.target_net.eval()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def save_model(self, name: str = '', directory: str = './saved_models/'):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        if name == '': # If no name was given
            name = f'CQLAgent_{self.num_optimizations}'
        torch.save(self.target_net.state_dict(), directory + name + '.pt')

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, split_transitions=True)

        # Predict next state
        with torch.no_grad():
            Q_targets_next = self.target_net(next_states.reshape(self.batch_size, -1)).detach().max(1)[0].unsqueeze(0)
            Q_targets = (rewards + (self.gamma * Q_targets_next * (1 - dones))).T

        Q_a_s = self.network(states.reshape(self.batch_size, -1))

        actions = actions.unsqueeze(-1).type(dtype=torch.int64)
        Q_expected = Q_a_s.gather(1, actions)

        cql1_loss = self.cql_loss(Q_a_s, actions)

        bellman_error = F.mse_loss(Q_expected, Q_targets)
        
        q1_loss = cql1_loss + 0.5 * bellman_error

        # Backpropagation with gradient clipping
        self.optimizer.zero_grad()
        q1_loss.backward()
        clip_grad_norm_(self.network.parameters(), 1.)
        self.optimizer.step()

        # Update target network
        soft_update(self.network, self.target_net, self.tau)

        # # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.num_optimizations += 1

    def reset(self):
        self.memory.reset()
        self.num_optimizations = 0
        self.network = DDQN(self.state_size, self.action_size, self.hidden_size, self.hidden_layers).apply(weights_init_).to(self.device)
        self.target_net = DDQN(self.state_size, self.action_size, self.hidden_size, self.hidden_layers).apply(weights_init_).to(self.device)

    def act(self, env: Env, deterministic: bool = False):
        state = env.get_state()

        # Epsilon-greedy policy
        if deterministic or random.random() > self.epsilon:
            state = torch.tensor(state, dtype=torch.float, device=self.device).reshape(-1).unsqueeze(0)
            self.network.eval()
            with torch.no_grad():
                action_values = self.network(state)
            self.network.train()
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
            action = int(action.squeeze())
        else:
            action = env.random_valid_action()
        return action

    def cql_loss(self, q_values, current_action):
        """Compute CQL loss for a batch of Q-values and actions."""

        logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
        q_a = q_values.gather(1, current_action)

        return (logsumexp - q_a).mean()
    
    def train_mode(self):
        if getattr(self, 'original_epsilon', None) == None:
            return
        # Restore original epsilon
        self.epsilon = self.original_epsilon
        # Delete temporary attribute
        delattr(self, 'original_epsilon')
    
    def eval_mode(self):
        # Save original epsilon in temporary attribute
        self.original_epsilon = self.epsilon
        # Set randomness to zero for evaluation
        self.epsilon = 0