""" Implements some utility functions. """

# Base libraries
import os
import random
from collections import deque, namedtuple
import numpy as np

# Machine Learning libraries
import torch
import torch.nn as nn

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class Memory(object):
    """
    Implementation of the memory class.
    """
    def __init__(self, max_capacity: int, min_capacity: int = 10000, device: torch.device = torch.device("cpu")):
        self.memory = deque([], maxlen=max_capacity)
        self.device = device
        self.min_capacity = min_capacity

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int, split_transitions: bool = False):
        minibatch = random.sample(self.memory, batch_size)

        if split_transitions:
            # Get individual elements of the namedtuple(s) as lists
            states, actions, rewards, next_states, dones = [], [], [], [], []
            for transition in minibatch:
                states.append(transition.state)
                actions.append(transition.action)
                rewards.append(transition.reward)
                next_states.append(transition.next_state)
                dones.append(transition.done)
            
            # Convert everything to stacked tensors with batch dimension being dim 0
            if type(states[0]) == torch.Tensor:
                if states[0].dim() == 2: # (HEIGHT x WIDTH)
                    states = torch.stack(states, dim=0).to(self.device).float()
                else: # (PLAYERS x HEIGHT x WIDTH)
                    states = torch.concat(states, dim=0).to(self.device)
            else: # Cast to tensor
                states = torch.tensor(states, device=self.device)
            if type(actions[0]) == torch.Tensor:
                actions = torch.concat(actions, dim=0).to(self.device)
            else: # Cast to tensor
                actions = torch.tensor(actions, device=self.device).reshape(batch_size, -1)
            if type(rewards[0]) == torch.Tensor:
                rewards = torch.concat(rewards, dim=0).to(self.device)
            else: # Cast to tensor
                rewards = torch.tensor(rewards, device=self.device).reshape(batch_size, -1)
            if type(next_states[0]) == torch.Tensor:
                if next_states[0].dim() == 2: # (HEIGHT x WIDTH)
                    next_states = torch.stack(next_states, dim=0).to(self.device).float()
                else: # (PLAYERS x HEIGHT x WIDTH)
                    next_states = torch.concat(next_states, dim=0).to(self.device)
            else: # Cast to tensor
                next_states = torch.tensor(next_states, device=self.device)
            if type(dones[0]) == torch.Tensor:
                dones = torch.concat(dones, dim=0).to(self.device)
            else: # Cast to tensor
                dones = torch.tensor(dones, device=self.device).reshape(batch_size, -1)
            
            return states.float(), actions, rewards, next_states.float(), dones
        else:
            return minibatch
    
    def __len__(self) -> int:
        return len(self.memory)
    
    def reset(self):
        self.memory.clear()
        
    def start_optimizing(self) -> bool:
        # Training starts when memory collected enough data.
        return self.__len__() >= self.min_capacity
    
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

def calculate_RTG(rewards):
    """
    Given an array of rewards, calculate the corresponding "Return-to-go" array!
    Example: If rewards are [1,2,3,4] then the result should be [10,9,7,4]
    because initially the total reward we get is 10 in this case.
    """
    RTGs = []
    for i in range(0,len(rewards)):
        # Sum up all the rewards occuring in the current timestep until the end!
        RTGs.append(sum(rewards[i:]))
    return RTGs

def get_opponent(player: int) -> int:
    """
    Given the current player ID, returns the ID of the opposing player.

    Args:
    player (int): ID of the current player.

    Returns:
    int: The ID of the opposing player.
    """
    return 3 - player


def select_highest_valid_action(action_values, valid_actions, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Selects the highest valid action for each sample in the batch on the specified device.

    Args:
    action_values (torch.Tensor): A tensor of shape (BATCH_SIZE, 7) with action values.
    valid_actions (list of lists): A nested list where each sublist contains valid actions as integers for each sample.
    device (torch.device): The device (CPU or CUDA) to perform computations on.

    Returns:
    torch.Tensor: A tensor of shape (BATCH_SIZE, 1) with the indices of the highest valid action for each sample.
    """

    BATCH_SIZE = action_values.shape[0]
        
    # Initialize a mask with very low values (so they don't get selected)
    mask = torch.full_like(action_values, -float('inf'), device=device)

    # Populate the mask with the actual values for valid actions
    for i in range(BATCH_SIZE):
        mask[i, valid_actions[i]] = action_values[i, valid_actions[i]]
    
    # Now select the action with the highest value that is valid
    selected_actions = torch.argmax(mask, dim=1, keepdim=True)

    return selected_actions