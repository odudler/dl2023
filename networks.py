import torch
import torch.nn as nn

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class DDQN(nn.Module):
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
    
class CNN(nn.Module):
    def __init__(self, state_size, action_size, layer_size, hidden_layers: int = 1):
        super(CNN, self).__init__()
        