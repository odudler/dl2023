import torch
import torch.nn as nn

class DDQN(nn.Module):
    def __init__(self, state_size, action_size, layer_size, hidden_layers: int = 1):
        super(DDQN, self).__init__()
        
        self.input_shape = state_size
        self.action_size = action_size

        self.activation = getattr(nn.functional, 'relu')

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_shape, layer_size))
        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(layer_size, layer_size))
        self.output_layer = nn.Linear(layer_size, action_size)
    
    def forward(self, s: torch.Tensor):
        # print(self.start, self.start.weight.device, flush=True)

        for layer in self.layers:
            s = self.activation(layer(s))
        return self.output_layer(s)
    
class CNN(nn.Module):
    def __init__(self, state_size, action_size, layer_size, hidden_layers: int = 1):
        super(CNN, self).__init__()

        self.input_shape = state_size
        self.action_size = action_size
        