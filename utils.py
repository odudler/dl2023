""" Implements some utility functions. """

# Base libraries
import os
import random
from collections import deque, namedtuple
import numpy as np
import matplotlib.pyplot as plt

# Machine Learning libraries
import torch
import torch.nn as nn

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class Memory(object):
    """
    Implementation of the memory class.
    """
    def __init__(self, max_capacity: int, min_capacity: int = 200, device: torch.device = torch.device("cpu")):
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
        
            for state, action, reward, next_state, done in minibatch:
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
            
            minibatch = [states, actions, rewards, next_states, dones]

            for i in range(len(minibatch)):
                minibatch[i] = torch.tensor(minibatch[i], dtype=torch.float32, device=self.device)
        
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
    
def soft_update(local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

# def calculate_RTG(rewards):
#     """
#     Given an array of rewards, calculate the corresponding "Return-to-go" array!
#     Example: If rewards are [1,2,3,4] then the result should be [10,9,7,4]
#     because initially the total reward we get is 10 in this case.
#     """
#     RTGs = []
#     for i in range(0,len(rewards)):
#         # Sum up all the rewards occuring in the current timestep until the end!
#         RTGs.append(sum(rewards[i:]))
#     return RTGs

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

def plot_rewards_steps(rewards, steps):

    rewards = np.array(rewards)
    steps = np.array(steps)
    
    if len(rewards) > 1000:
        indices = np.linspace(0, len(rewards) - 1, 1000, dtype=int)
        rewards = rewards[indices]
        steps = steps[indices]

    print("Plotting Episode rewards and steps...")
    plt.figure(figsize=(15, 4))
    plt.title('Episode rewards and episode steps')
    plt.xlabel('Episode')
    plt.plot(steps, color='blue', label='Episode steps')
    plt.plot(rewards, color='red', label='Episode rewards')
    plt.legend()
    
    plt.show()
