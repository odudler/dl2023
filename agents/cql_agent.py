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
from utils import Memory, soft_update
from env import Env

class CQLAgent(Agent):
    """
    Implementation of CQL DQN agent.
    For more information see https://github.com/BY571/CQL/tree/main/CQL-DQN. 
    """
    def __init__(self, env: Env, state_size: int = 42, action_size: int = 7, hidden_size: int = 64, hidden_layers: int = 3, batch_size: int = 4, 
                 epsilon_max: float = 1.0, epsilon_min: float = 0.1, epsilon_decay: float = 0.999,
                 device: torch.device = torch.device("cpu"), options: Union[None, dict] = None):
        super(CQLAgent, self).__init__(learning=True)

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
            self.network = DDQN(self.state_size, self.action_size, self.hidden_size, self.hidden_layers).to(self.device)
            self.target_net = DDQN(self.state_size, self.action_size, self.hidden_size, self.hidden_layers).to(self.device)
            self.target_net.load_state_dict(self.network.state_dict())

        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)

    def load_model(self, loadpath: str):
        self.network.load_state_dict(torch.load(loadpath, map_location=self.device))
        self.network.eval()
        self.target_net.load_state_dict(torch.load(loadpath, map_location=self.device))
        self.target_net.eval()

    def save_model(self, name: str = '', directory: str = './saved_models/'):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        if name == '': # If no name was given
            name = f'CQLAgent_{self.num_optimizations}'
        torch.save(self.target_net.state_dict(), directory + name + '.pt')
        
    def remember(self, state: list, action: list, reward: list, next_state: list, done: list):
        self.memory.push(state, action, reward, next_state, done)

    def optimize_model(self):
        if not self.memory.start_optimizing():
            return
        
        self.network.train()
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, split_transitions=True)
        move_validity = next_states[:, 0, :] == 0
        # Predict next state
        with torch.no_grad():
            Q_targets_next = self.target_net(next_states.reshape(self.batch_size, -1))
            Q_targets_next = torch.where(move_validity, Q_targets_next, -1e7)
            Q_targets_next = Q_targets_next.detach().max(1)[0].unsqueeze(0)
            Q_targets = (rewards + (self.gamma * Q_targets_next * (1 - dones))).T # TODO: Modify this, dones is not boolean! dones takes values -1, 0, 1, 2

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
        self.network = DDQN(self.state_size, self.action_size, self.hidden_size, self.hidden_layers).eval().to(self.device)
        self.target_net = DDQN(self.state_size, self.action_size, self.hidden_size, self.hidden_layers).eval().to(self.device)
        self.target_net.load_state_dict(self.network.state_dict())

    def act(self, state: list, **kwargs):

        # Epsilon-greedy policy
        if kwargs['deterministic'] or random.random() > self.epsilon:
            state = torch.tensor(state, dtype=torch.float, device=self.device).reshape(1, -1)
            self.network.eval()
            with torch.no_grad():
                q_values = self.network(state)
            self.network.train()
            action = torch.argmax(q_values, dim=1)
            action = int(action)
        else:
            action = self.env.random_valid_action()
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