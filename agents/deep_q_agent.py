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
from networks import FCNN, CNN
from utils import Memory
from env import Env

class DeepQAgent(Agent):
    """
    Implementation of the deep Q Agent, performing Q learning with neural networks.
    """
    def __init__(self, env: Env, state_size: int = 42, action_size: int = 7, hidden_size: int = 64, hidden_layers: int = 2, batch_size: int = 128,
                 epsilon_max: float = 1.0, epsilon_min: float = 0.01, epsilon_decay: float = 0.999, network_type: str = 'FCNN',
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
        self.memory = Memory(max_capacity=10000, min_capacity=200, device=self.device)
        
        # Parameters
        self.gamma = 0.9 # Discount rate
        self.lr = 0.01 # Optimizer learning rate
        
        self.num_optimizations = 0
        self.network_type = network_type

        if type(options) == dict:
            self.options = options
            if type(self.options['weights_init']) == DeepQAgent: # Initialize with weights from passed model
                self.network = copy.deepcopy(self.options['weights_init'].network).eval().to(self.device)
            else:
                raise ValueError(f'Cannot copy weigths to new model, invalid model type {type(self.options["weights_init"])}.')
        else: # No additional options passed, initialize new model
            assert network_type in ['FCNN', 'CNN'], "Network type has to be one of ['FCNN', 'CNN']"
            if network_type == 'FCNN':
                self.network = FCNN(self.state_size, self.action_size, self.hidden_size, self.hidden_layers).eval().to(self.device)
            elif network_type == 'CNN': # CNN
                self.network = CNN().eval().to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

    def load_model(self, loadpath: str):
        self.network.load_state_dict(torch.load(loadpath, map_location=self.device))
        self.network.eval()

    def save_model(self, name: str = '', directory: str = './saved_models/'):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        if name == '': # If no name was given
            name = f'DeepQAgent_{self.network_type}_{self.num_optimizations}'
        torch.save(self.network.state_dict(), directory + name + '.pt')
        print(f"Model was saved in {directory} as {name}.pt")
        
    def remember(self, state: list, action: list, reward: list, next_state: list, done: list):
        self.memory.push(state, action, reward, next_state, done)

    def optimize_model(self):
        # Only start optimizing once memory has reached min_capacity
        if not self.memory.start_optimizing():
            return
        
        self.network.train()
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
            next_q_values = self.network(next_states).detach() # Detach since no gradient calc needed
            next_q_values = torch.where(move_validity, next_q_values, -1e7).max(dim=1, keepdim=True)[0] # Get max Q-Values for the next_states.
            expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        q_values = self.network(states).gather(dim=1, index=actions) # Get Q-Values for the actions
        
        loss = self.criterion(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.num_optimizations += 1

    def reset(self):
        self.memory.reset()
        self.num_optimizations = 0
        if self.network_type == 'FCNN':
            self.network = FCNN(self.state_size, self.action_size, self.hidden_size, self.hidden_layers).eval().to(self.device)
        elif self.network_type == 'CNN':
            self.network = CNN().eval().to(self.device)
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