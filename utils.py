"""Implements some utility functions."""

# Base libraries
import os
import random
from collections import deque, namedtuple
import numpy as np

# Machine Learning libraries
import torch
import torch.nn as nn

# Local imports
from agent_interface import Agent
from env import Env

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'mask'))

"""Implement memory class"""
class Memory(object):
    def __init__(self, max_capacity):
        self.memory = deque([], maxlen=max_capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    def reset(self):
        self.memory.clear()
    
def seed_everything(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)