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
from torch.nn.utils import clip_grad_norm_

# Local imports
from agents.agent_interface import Agent
from networks import FCNN, CNN
from utils import Memory, soft_update
from env import Env

class CQLAgent(Agent):
    """
    Implementation of CQL agent. The network type can be chosen, either 'FCNN' or 'CNN'.
    Adapted from https://github.com/BY571/CQL/tree/main/CQL-DQN. 
    """
    def __init__(self, env: Env, state_size: int = 42, action_size: int = 7, hidden_size: int = 64, hidden_layers: int = 2, batch_size: int = 128,
                 epsilon_max: float = 1.0, epsilon_min: float = 0.01, epsilon_decay: float = 0.999, network_type: str = 'FCNN',
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
        self.memory = Memory(max_capacity=10000, min_capacity=200, device=self.device)
        
        # Parameters
        self.gamma = 0.9 # Discount rate
        self.tau = 1e-3 # Soft update param
        self.lr = 0.01 # Optimizer learning rate
        
        self.num_optimizations = 0
        self.network_type = network_type

        if type(options) == dict:
            self.options = options
            if type(self.options['weights_init']) == CQLAgent: # Initialize with weights from passed model
                self.network = copy.deepcopy(self.options['weights_init'].network).eval().to(self.device)
                self.target_net = copy.deepcopy(self.options['weights_init'].target_net).eval().to(self.device)
            else:
                raise ValueError(f'Cannot copy weigths to new model, invalid model type {type(self.options["weights_init"])}.')
        else: # No additional options passed, initialize new model
            assert network_type in ['FCNN', 'CNN'], "Network type has to be one of ['FCNN', 'CNN']"
            if network_type == 'FCNN':
                self.network = FCNN(self.state_size, self.action_size, self.hidden_size, self.hidden_layers).eval().to(self.device)
                self.target_net = FCNN(self.state_size, self.action_size, self.hidden_size, self.hidden_layers).eval().to(self.device)
                self.target_net.load_state_dict(self.network.state_dict())
            elif network_type == 'CNN': # CNN
                self.network = CNN().eval().to(self.device)
                self.target_net = CNN().eval().to(self.device)
                self.target_net.load_state_dict(self.network.state_dict())
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

    def load_model(self, loadpath: str):
        self.network.load_state_dict(torch.load(loadpath, map_location=self.device))
        self.network.eval()
        self.target_net.load_state_dict(torch.load(loadpath, map_location=self.device))
        self.target_net.eval()

    def save_model(self, name: str = '', directory: str = './saved_models/'):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        if name == '': # If no name was given
            name = f'CQLAgent_{self.network_type}_{self.num_optimizations}'
        torch.save(self.network.state_dict(), directory + name + '.pt')
        print(f"Model was saved in {directory} as {name}.pt")
        
    def remember(self, state: list, action: list, reward: list, next_state: list, done: list):
        self.memory.push(state, action, reward, next_state, done)

    def optimize_model(self):
        # Only start optimizing once memory has reached min_capacity
        if not self.memory.start_optimizing():
            return
        
        self.network.train()
        self.target_net.train()
        # Get samples from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, split_transitions=True)
        move_validity = next_states[:, 0, :] == 0 # For full columns, q-values should be -inf (-1e7) since this move will be invalid (See below)
        states = states.reshape(self.batch_size, -1).to(self.device)
        actions = actions.to(torch.int64).reshape(self.batch_size, -1).to(self.device)
        rewards = rewards.reshape(self.batch_size, -1).to(self.device)
        next_states = next_states.reshape(self.batch_size, -1).to(self.device)
        dones[dones >= 0] = 1 # Game finished (finished == 0, 1, 2)
        dones[dones < 0] = 0 # Game did not finish (finished == -1)
        dones = dones.reshape(self.batch_size, -1).to(self.device)
        
        with torch.no_grad():
            Q_targets_next = self.target_net(next_states).detach()
            Q_targets_next = torch.where(move_validity, Q_targets_next, -1e7).max(dim=1, keepdim=True)[0] # Get max Q-Values for the next_states.
            Q_targets = rewards + (1 - dones) * self.gamma * Q_targets_next

        Q_a_s = self.network(states)
        Q_expected = Q_a_s.gather(dim=1, index=actions)

        cql1_loss = self.cql_loss(Q_a_s, actions)

        bellman_error = self.criterion(Q_expected, Q_targets)
        
        q1_loss = cql1_loss + 0.5 * bellman_error

        # Backpropagation with gradient clipping
        self.optimizer.zero_grad()
        q1_loss.backward()
        clip_grad_norm_(self.network.parameters(), 1.)
        self.optimizer.step()
        
        # Update target network
        soft_update(self.network, self.target_net, self.tau)
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.num_optimizations += 1

    def reset(self):
        self.memory.reset()
        self.num_optimizations = 0
        if self.network_type == 'FCNN':
            self.network = FCNN(self.state_size, self.action_size, self.hidden_size, self.hidden_layers).eval().to(self.device)
            self.target_net = FCNN(self.state_size, self.action_size, self.hidden_size, self.hidden_layers).eval().to(self.device)
            self.target_net.load_state_dict(self.network.state_dict())
        elif self.network_type == 'CNN':
            self.network = CNN().eval().to(self.device)
            self.target_net = CNN().eval().to(self.device)
            self.target_net.load_state_dict(self.network.state_dict())
        else:
            raise ValueError("Network type needs to be one of ['FCNN', 'CNN'].")

    def act(self, state: list, **kwargs) -> int:
        """
        Returns the best estimated action based on the current state of the board.

        Args:
            state (list): The current state of the game in appropriate form.
            kwargs['deterministic'] (bool): If true, sets the probability of choosing a random action to zero.

        Returns:
            int: The current estimate of the reward-maximizing action or a random valid action.
        """
        # Epsilon-greedy policy
        if kwargs['deterministic'] or random.random() > self.epsilon:
            state = torch.tensor(state, dtype=torch.float, device=self.device).reshape(1, -1)
            self.network.eval()
            with torch.no_grad():
                q_values = self.network(state)
            self.network.train()
            best_actions = torch.argsort(q_values, dim=1, descending=True).squeeze().tolist() # best actions are ordered from best to worst
            valid_actions = self.env.field.get_valid_cols()
            best_valid_actions = [a for a in best_actions if a in valid_actions]
            action = best_valid_actions[0]
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