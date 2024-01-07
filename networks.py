import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        torch.nn.init.constant_(m.bias, 0)

class FCNN(nn.Module):
    """
    Fully connected neural network. This class implements a neural network with a variable number of hidden layers and hidden units.
    Used for CQLAgent and DeepQAgent.
    """
    def __init__(self, state_size, action_size, layer_size, hidden_layers):
        super(FCNN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_size, layer_size))       # Input layer
        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(layer_size, layer_size))   # Hidden layers
        self.output_layer = nn.Linear(layer_size, action_size)      # Output layer
        
        self.activation = nn.functional.relu
        self.apply(weights_init_)
    
    def forward(self, s: torch.Tensor):

        for layer in self.layers:
            s = self.activation(layer(s))
        return self.output_layer(s)

# Network adapted from https://www.kaggle.com/code/wrinkledtime/reinforce-rl/notebook
class CNN(nn.Module):
    """
    This class implements a small convolutional neural network (CNN).
    Used for CQLAgent and DeepQAgent
    """
    
    def __init__(self):
        """
        Generates a small CNN with one 2d convolutional layer and two linear layers mapping to the output.
        We assume fixed board size 6x7 with 7 possible actions.

        Args:
            state_size (int): Size of state.
            action_size (int): Size of action.

        """
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=42, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(840, 42) # (6-3+1)*(7-3+1)*42 = 840
        self.fc2 = nn.Linear(42, 7)
        self.apply(weights_init_)
        
    def forward(self, z):
        """
        Forward pass. Here we assume that the agent has ID = 1, and the opponent ID = 2.
        Input tensor will be transformed to a tensor of shape (batch_size, 6, 7, 2). The last dimension holds boolean indicators (0 and 1), indicating where each player has his coins.
        
        Args:
            z (torch.Tensor): Input, state with shape (batch_size, 42)

        Returns:
            actions (torch.Tensor): Shape (batch_size, 1)
        """
        # Input z should be a tensor of shape (batch_size, 42), reshape to (batch_size, 6, 7)
        z = z.reshape(-1, 6, 7)
        batch_size = z.shape[0]

        # Channel dimension is second dimension for Conv2d
        x = torch.zeros((batch_size, 2, 6, 7), dtype=torch.int32)
        x[:, 0, :, :] = (z == 1)
        x[:, 1, :, :] = (z == 2)
        x = x.to(dtype=torch.float32)
        
        x = F.relu(self.conv1(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        actions = F.softmax(self.fc2(x), dim=-1)
        return actions
        