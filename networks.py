import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        torch.nn.init.constant_(m.bias, 0)

class DDQN(nn.Module):
    """
    This class implements a neural network with a variable number of hidden layers and hidden units.
    Used for CQLAgent and DeepQAgent.
    """
    def __init__(self, state_size, action_size, layer_size, hidden_layers: int = 1):
        super(DDQN, self).__init__()

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
    
    def __init__(self, input_shape: Union[torch.Tensor, torch.Size], output_shape: Union[int, torch.Size, torch.Tensor]):
        """Generates a small CNN with one 2d convolutional layer and two linear layers mapping to the output. This network assumes as input shape a 'one-hot' encoded board, i.e. 2 boards for two players with boolean indicators (0 and 1) indicating where each player has his coins. 

        Args:
            input_shape (Union[torch.Tensor, torch.Size]): Either (NUM_PLAYERS x NUM_ROWS x NUM_COLUMNS) or (BATCH_SIZE x NUM_PLAYERS x NUM_ROWS x NUM_COLUMNS)
            output_shape (Union[int, torch.Size, torch.Tensor]): Output shape of the network. (NUM_ACTIONS)

        """
        super(CNN, self).__init__()

        if len(input_shape) == 3:
            P, H, W = input_shape # Assumes dimensions are (NUM_PLAYERS x NUM_ROWS x NUM_COLUMNS)
        elif len(input_shape) == 4:
            _, P, H, W = input_shape # Assume first dimension as batch dimension
        else:
            raise ValueError("Wrong number of dimensions. CNN assumes either a 3- or 4-dimensional input shape. See documentation.")
        
        assert H >= 3, f"Height of input has to be >2, but was {H}."
        assert W >= 3, f"Width of input has to be >2, but was {W}"

        self.conv1 = nn.Conv2d(in_channels=P, out_channels=H*W, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(H * W * (H - 2) * (W - 2), H * W) # Map to board shape
        self.fc2 = nn.Linear(H * W, output_shape) # Map board to action shape
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        actions = F.softmax(self.fc2(x), dim=-1)
        return actions
        