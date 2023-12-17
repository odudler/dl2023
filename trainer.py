# Base libraries
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, Any
import numpy as np
import pandas as pd
import random
from typing import Union

# ML libraries
import torch

# Local imports
from board import ConnectFourField
from env import Env
from agents.agent_interface import Agent
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent
from agents.minimax_agent_old import OldMinimaxAgent
from agents.deep_q_agent import DeepQAgent
from agents.cql_agent import CQLAgent
import utils

class Trainer():
    """
    Trainer class.
    """
    def __init__(self, env: Union[Env, None],
                 agent: Agent,
                 opponent: Agent,
                 options: Union[dict, None],
                 agent_id: int = 1,
                 opponent_id: int = 2,
                 modes: list = ['TRAIN', 'EVAL'],
                 num_episodes: Union[int, dict] = {'TRAIN': 1000, 'EVAL': 100},
                 device: str = 'cpu',
                 verbose: bool = False,
                 ):
        # Environment
        self.env = env if env else Env()
        # Agent
        self.agent = agent
        # Opponent
        self.opponent = opponent
        # Hyperparameters
        assert(all([id in [1, 2] for id in [agent_id, opponent_id]]))
        self.AGENT = agent_id
        self.OPPONENT = opponent_id
        assert(all(mode in ['TRAIN', 'EVAL', 'TEST'] for mode in modes))
        self.MODES = modes
        if isinstance(num_episodes, int):
            assert(num_episodes > 0)
        else:
            assert(value > 0 for value in num_episodes.values())
        self.NUM_EPISODES = num_episodes
        assert(device in ['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
        self.DEVICE = device
        self.VERBOSE = verbose
        # Parse options
        if options: self.parse_options(options)

    def parse_options(self, options: dict):
        """
        Parses the dictionary containing the options and sets default values where needed.
        """
        keys = options.keys()
        if 'BOOTSTRAP_EPISODES' in keys:
            assert(type(options['BOOTSTRAP_EPISODES']) == int and options['BOOTSTRAP_EPISODES'] >= 0)
            self.BOOTSTRAP_EPISODES = options['BOOTSTRAP_EPISODES']
        if 'UPDATE_OPPONENT' in keys:
            assert(type(options['UPDATE_OPPONENT']) == bool)
            self.UPDATE_OPPONENT = options['UPDATE_OPPONENT']
            if self.UPDATE_OPPONENT:
                if 'OPPONENT_UPDATE_FREQUENCY' in keys:
                    assert(type(options['OPPONENT_UPDATE_FREQUENCY']) == int and options['OPPONENT_UPDATE_FREQUENCY'] > 0)
                    self.OPPONENT_UPDATE_FREQUENCY = options['OPPONENT_UPDATE_FREQUENCY']
                else:
                    self.OPPONENT_UPDATE_FREQUENCY = 100 # Default value
        if 'DECAY_RANDOMNESS_OPPONENT' in keys:
            assert(type(self.opponent) in [MinimaxAgent])
            assert(type(options['DECAY_RANDOMNESS_OPPONENT']) == bool)
            self.DECAY_RANDOMNESS_OPPONENT = options['DECAY_RANDOMNESS_OPPONENT']
            if self.DECAY_RANDOMNESS_OPPONENT:
                if 'DECAY_RANDOMNESS_FREQUENCY' in keys:
                    assert(type(options['DECAY_RANDOMNESS_FREQUENCY']) == int and options['DECAY_RANDOMNESS_FREQUENCY'] > 0)
                    self.DECAY_RANDOMNESS_FREQUENCY = options['DECAY_RANDOMNESS_FREQUENCY']
                else:
                    self.DECAY_RANDOMNESS_FREQUENCY = 200 # Default value
        if 'REPLACE_FOR_EVALUATION' in keys:
            assert(type(options['REPLACE_FOR_EVALUATION']) == bool)
            self.REPLACE_FOR_EVALUATION = options['REPLACE_FOR_EVALUATION']
            if 'REPLACE_FOR_EVALUATION_BY' in keys:
                assert(isinstance(options['REPLACE_FOR_EVALUATION_BY'], Agent))
                self.REPLACE_FOR_EVALUATION_BY = options['REPLACE_FOR_EVALUATION_BY']
            else:
                self.REPLACE_FOR_EVALUATION_BY = RandomAgent() # Default value
        # Define additional options here
        # TODO: Option for printing the last n games or for periodically printing a game

    def train(self):
        """
        Implements the game loop
        """
        for mode in self.MODES:
            # Reset score counter
            p1_score = 0
            p2_score = 0
            # Keep track of total number of turns and invalid turns
            if self.VERBOSE:
                turns = 0
                invalid = 0
            else:
                turns = None
                invalid = None

            # Play all episodes
            for i in range(1, self.NUM_EPISODES[mode]):
                # Clean up terminal line 
                if i % 100 == 0: print('\r                                                                                                                       ', end='')
                # Print current episode
                print(f'\r{mode}: Running episode {i} of {self.NUM_EPISODES[mode]}. Current win ratio of AGENT is {p1_score / i:.2%}.', end='')
                # Make it random who starts
                agent_start = random.choice([True, False])
                # Run one episode of the game and update running variables
                finished, turns, invalid = self.play_episode(mode, agent_start, turns, invalid)
                # Update scores with winner
                if finished == 1:
                    p1_score += 1
                elif finished == 2:
                    p2_score += 1
                # Optional periodic updates
                self.perform_periodic_updates(mode=mode, episode=i)
                # Reset board to empty
                self.env.reset()
            # Current MODE done, print on new line
            print(f'\n{mode}: Average turns per episode', turns / self.NUM_EPISODES[mode])
            print(f'{mode}: Average invalid moves per episode', invalid / self.NUM_EPISODES[mode])
            print('\n')
            
    def play_episode(self, mode: str = 'TRAIN', agent_start: bool = True, turns: Union[int, None] = None, invalid: Union[int, None] = None):
        """
        Let agent and opponent play one episode
        """

        # Initialize episodic variables
        finished = -1

        # Run one game
        while finished == -1:
            ######################
            # Agent makes a turn #
            ######################
            if agent_start: # Agent starts the game
                # Get current state of the game
                state = self.env.get_state() if not type(self.agent) in [MinimaxAgent, DeepQAgent] else None
                state_var = self.env.get_state()
                # Predict best next action based on this state
                action = self.agent.act(state if state else self.env)
                # Execute action
                valid, reward, finished = self.env.step(action, self.AGENT)
                # Print and track stuff for debugging
                if self.VERBOSE:
                    if turns != None: turns += 1
                    if invalid != None and valid == -1: invalid += 1
                # Perform learning step for a learning opponent
                if self.agent.learning:
                    next_state = self.env.get_state()
                    self.agent.remember(state_var, action, reward, next_state, finished)
                    self.agent.optimize_model()
                # End episode if somebody won or it is a tie
                if finished != -1: break
            else: # Agent doesn't start the game
                agent_start = True
            #########################
            # Opponent makes a turn #
            #########################
            # Get current state of the game
            state = self.env.get_state() if not type(self.agent) in [MinimaxAgent, DeepQAgent] else None
            # Predict best next action based on this state
            action = self.opponent.act(state if state else self.env)
            # Execute action
            valid, _, finished = self.env.step(action, self.OPPONENT)
            # Print and track stuff for debugging
            if self.VERBOSE:
                if turns != None: turns += 1
                if invalid != None and valid == -1: invalid += 1
            # End episode if somebody won or it is a tie
            if finished != -1: break
        # Return updated variables
        if self.VERBOSE:
            return finished, turns, invalid
        else: return finished, None, None

    def perform_periodic_updates(self, mode: str, episode: int):
        assert(episode >= 0)
        assert(mode in ['TRAIN', 'EVAL', 'TEST'])

        # Update opponent periodically
        if getattr(self, 'UPDATE_OPPONENT', None):
            BOOTSTRAP_EPISODES = getattr(self, 'BOOTSTRAP_EPISODES', None)
            if (BOOTSTRAP_EPISODES and episode > BOOTSTRAP_EPISODES and episode % self.OPPONENT_UPDATE_FREQUENCY == 0) or \
            (not BOOTSTRAP_EPISODES and episode % self.OPPONENT_UPDATE_FREQUENCY == 0):
                # Update opponent with current version of the agent
                agent_class = type(self.agent)
                assert(agent_class in [DeepQAgent, CQLAgent])
                self.opponent = agent_class(epsilon_max=0.1, epsilon_min=0.1, device=self.DEVICE, options={'weights_init': self.agent})
        
        # Decay randomness of opponent
        if getattr(self, 'DECAY_RANDOMNESS_OPPONENT', None):
            if episode % self.DECAY_RANDOMNESS_FREQUENCY == 0:
                assert(type(self.opponent) is MinimaxAgent)
                self.opponent.decay_epsilon() # TODO: Make decay rate and min randomness hyperparameters as well
        
        # Replace opponent for evaluation
        if mode in ['EVAL', 'TEST']:
            if getattr(self, 'REPLACE_FOR_EVALUATION', None):
                self.opponent = self.REPLACE_FOR_EVALUATION_BY