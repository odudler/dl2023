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
from networks import DDQN, CNN
from utils import Memory, soft_update, select_highest_valid_action, weights_init_
from env import Env

class CQLAgent(Agent):
    def __init__(self, env: Env, state_size: Union[int, torch.Size, torch.Tensor] = 42, action_size: int = 7, hidden_size: int = 64, hidden_layers: int = 3, batch_size: int = 32, 
                 epsilon_max: float = 1.0, epsilon_min: float = 0.1, epsilon_decay: float = 0.999, network_type: str = 'DDQN',
                 device: torch.device = torch.device("cpu"), options: Union[None, dict] = None):
        super(CQLAgent, self).__init__(learning=True)
        
        self.device = device
        self.network_type = network_type
        self.env = env

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.memory = Memory(max_capacity=1000, device=self.device, min_capacity=self.batch_size) # Replay memory

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
            assert network_type in ["DDQN", "CNN"], "Network type has to be one of ['DDQN', 'CNN']"
            if network_type == "DDQN":
                self.network = DDQN(state_size, action_size, hidden_size, hidden_layers).apply(weights_init_).to(self.device)
                self.target_net = DDQN(state_size, action_size, hidden_size, hidden_layers).apply(weights_init_).to(self.device)
            elif network_type == "CNN": # CNN
                self.network = CNN(state_size, action_size).apply(weights_init_).to(self.device)
                self.target_net = CNN(state_size, action_size).apply(weights_init_).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)

    def load_model(self, loadpath):
        self.network.load_state_dict(torch.load(loadpath, map_location=torch.device(self.device)))
        self.network.eval()

        self.target_net.load_state_dict(torch.load(loadpath, map_location=torch.device(self.device)))
        self.target_net.eval()
    
    def remember(self, state: torch.Tensor, action: torch.Tensor, reward, next_state: torch.Tensor, done):
        self.memory.push(state, action, reward, next_state, done)

    def save_model(self, name: str = '', directory: str = './saved_models/'):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        if name == '': # If no name was given
            name = f'CQLAgent_{self.network_type}_{self.num_optimizations}'
        torch.save(self.target_net.state_dict(), directory + name + '.pt')

    def optimize_model(self):
        # Only start optimizing once memory has reached batch size
        if not self.memory.start_optimizing():
            return
        # Get samples from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, split_transitions=True)

        # Predict next state
        with torch.no_grad():
            if type(self.target_net) == DDQN:
                Q_targets_next = self.target_net(next_states.flatten(start_dim=1)).detach().max(1)[0].unsqueeze(1) # (B x 1)
            else:
                Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1) # (B x 1)
            # Q_targets_next = self.target_net(next_states.reshape(self.batch_size, -1)).detach().max(1)[0].unsqueeze(0)
            Q_targets = (rewards + (self.gamma * Q_targets_next * (1 - dones)))

        # Q_a_s = self.network(states.reshape(self.batch_size, -1))
        if type(self.target_net) == DDQN:
            Q_a_s = self.network(states.flatten(start_dim=1))
        else:
            Q_a_s = self.network(states)

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

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.num_optimizations += 1

    def reset(self):
        self.memory.reset()
        self.num_optimizations = 0
        if self.network_type == 'DDQN':
            self.network = DDQN(self.state_size, self.action_size, self.hidden_size, self.hidden_layers).apply(weights_init_).to(self.device)
            self.target_net = DDQN(self.state_size, self.action_size, self.hidden_size, self.hidden_layers).apply(weights_init_).to(self.device)
        elif self.network_type == 'CNN':
            self.network = CNN(self.state_size, self.action_size).apply(weights_init_).to(self.device)
            self.target_net = CNN(self.state_size, self.action_size).apply(weights_init_).to(self.device)
        else:
            raise ValueError("Network type needs to be one of ['DDQN', 'CNN'].")


    def act(self, state: torch.Tensor, deterministic: bool = False, *args) -> Union[torch.Tensor, int]:
        """Returns the best estimated action based on the current state of the board.

        Args:
            env (Env): The current environment the agent is operating in.
            state (torch.Tensor): The current state of the game in appropriate form.
            deterministic (bool, optional): If true, sets the probability of choosing a random action to zero. Defaults to False.

        Returns:
            Union[torch.Tensor, int]: The current estimate of the reward-maximizing action or a random valid action.
        """
        # Epsilon-greedy policy
        if deterministic or random.random() > self.epsilon:
            self.network.eval()
            with torch.no_grad():
                if self.network_type == 'DDQN':
                    if state.dim() == 2:
                        state = state.unsqueeze(0)
                    action_values = self.network(state.to(self.device).float().flatten(start_dim=1))
                else:
                    action_values = self.network(state.to(self.device).float())
                valid_cols = torch.tensor(self.env.field.get_valid_cols(), device=self.device)
            
            action = select_highest_valid_action(action_values, valid_cols, device=self.device).cpu()
            self.network.train()
        else:
            if state.dim() == 3:
                action = torch.tensor([self.env.random_valid_action() for _ in range(state.shape[0])])
            else:
                action = torch.tensor([self.env.random_valid_action()])
            while action.dim() < 2:
                action = action.unsqueeze(0)
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