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
from agents.agent_interface import Agent
from env import Env

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'mask'))

"""Implement memory class"""
class Memory(object):
    def __init__(self, max_capacity):
        self.memory = deque([], maxlen=max_capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int, split_transitions: bool = False):
        minibatch = random.sample(self.memory, batch_size)

        if split_transitions:
            states, actions, rewards, next_states, dones = [], [], [], [], []
        
            for state, action, reward, next_state, done in minibatch:
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
            
            minibatch = [states, actions, rewards, next_states, dones]

            for i in range(len(minibatch)):
                minibatch[i] = torch.tensor(minibatch[i], dtype=torch.float32)
        
        return minibatch
    
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
    
def soft_update(local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

#given an array of rewards, calculate the corresponding "Return-to-go" array!
#Example: if rewards are [1,2,3,4] then the result should be [10,9,7,4]
#Becaues initially the reward we totally get is 10 in this case etc.
def calculate_RTG(rewards):
    RTGs = []
    for i in range(0,len(rewards)):
        #Sum up all the rewards occuring in the current timestep until the end!
        RTGs.append(sum(rewards[i:]))
    return RTGs