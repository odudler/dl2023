# Standard libraries
from collections import namedtuple, deque
from typing import Union

# Deep Learning libraries
import torch
from torch import nn
import torch.nn.functional as F

# Local imports
from env import Env
from agent_interface import Agent
from random_agent import RandomAgent
from deep_q_agent import DeepQAgent
import utils

class Trainer():
    def __init__(self, player1: Agent = DeepQAgent(), player2: Agent = RandomAgent(),
                 num_episodes: int = 100, policy_net_update_frequency: int = 10, 
                 soft_update_param: float = 1.0, random_bootstrapping: bool = True, player_start: bool = True, 
                 model_dir: str = '/default/path/to/models', seed: int = 42, verbose: bool = False):
        assert(soft_update_param >= 0.0 and soft_update_param <= 1.0)

        # Hyperparameters / Options
        self.episodes = num_episodes
        self.policy_net_update_frequency = policy_net_update_frequency
        self.soft_update_param = soft_update_param
        self.bootstrapping = random_bootstrapping
        self.verbose = verbose

        # Players
        self.p1 = player1
        self.p1_score = 0
        self.p2 = player2
        self.p2_score = 0

        # Environment
        self.env = Env(starts = 1 if player_start else 2, seed=seed)
        self.model_path = model_dir

    
    def train(self):
        for i in range(self.episodes):
            # Set up game
            observation = self.env.reset() # TODO: Make reset return state
            done = self.env.field.is_finished()
            
            # Play game
            while not done:
                if self.env.turn == 1:
                    # Player 1 makes move
                    action = self.p1.act(observation)
                    observation_, reward, done, info = self.env.step(action, player=1)
                    self.p1.optimize_model() # TODO
                    observation = observation_
                    self.env.turn = 2
                elif self.env.turn == 2:
                    # Player 2 makes move
                    action = self.p2.act(observation)
                    observation_, reward, done, info = self.env.step(action, player=2)
                    observation = observation_
                    self.env.turn = 1
                else:
                    raise ValueError(f'Invalid turn variable. Was {self.env.turn}, should be one of [1, 2]')
            
            # Game finished, print results
            episode_str = f'Winner of episode {i} was {self.env.winner}.'
            if self.env.winner == 1:
                self.p1_score += 1
            elif self.env.winner == 2:
                self.p2_score += 1
            else:
                episode_str = f'Episode {i} ended in a draw.'
            print(episode_str + f' P1 has {self.p1_score} wins, P2 has {self.p2_score} wins, and there were {i - self.p1_score - self.p2_score} draws.')
            if self.verbose:
                print('End state of the game was:')
                self.env.render_console()

            # Optimize player model
            if i % self.policy_net_update_frequency == 0:
                # TODO: Model update
                pass

            # Replace random agent
            if self.bootstrapping and i % self.policy_net_update_frequency == 0:
                self.p2 = self.p1

        # TODO: Create dirs if they don't exist yet, save model in dir
        self.p1.save_model('/path/to/save/model')
