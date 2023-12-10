"Implementation of the Decision Transformer Agent"
from .agent_interface import Agent

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Using Decision Transformer implementation from https://huggingface.co/docs/transformers/model_doc/decision_transformer
from transformers import DecisionTransformerConfig, DecisionTransformerModel

from utils import Memory

class DT_Agent(Agent):
    def __init__(self, env, state_size: int = 42, action_size: int = 7, batch_size: int = 4, learning_rate = 0.0001):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Store sequences (R, s, a, t) in memory
        self.memory = Memory(max_capacity=1000)
        
        self.dt_model_config = DecisionTransformerConfig()
        self.dt_model = DecisionTransformerModel(self.dt_model_config)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        
    def load_model(self, loadpath):
        self.model.load_state_dict(torch.load(loadpath))
        self.model.eval()

    def save_model(self, savepath):
        torch.save(self.model.state_dict(), savepath)

    def optimize_model(self):
        # Train decision transformer model using data in the memory (?)
        pass

    def reset(self):
        self.memory.reset()

    def act(self, state):
        # Act based on decision transformer output
        state, actions, rewards, target_return, timesteps, attention_mask = 
        with torch.no_grad():
            state_preds, action_preds, return_preds = model(
                states=state,
                actions=actions,
                rewards=rewards,
                returns_to_go=target_return,
                timesteps=timesteps,
                attention_mask=attention_mask,
                return_dict=False,
            )
        return self.env.random_valid_action()