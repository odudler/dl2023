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
        # Helper variables
        self.num_optimizations = 0

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
        if 'AUTOSAVE' in keys:
            assert(type(options['AUTOSAVE']) == bool)
            self.AUTOSAVE = options['AUTOSAVE']
            if self.AUTOSAVE:
                self.times_saved = 0 # Running variable
                if 'AUTOSAVE_TYPE' in keys:
                    assert(isinstance(options['AUTOSAVE_TYPE'], str) and options['AUTOSAVE_TYPE'] in ['NUM_OPTIMIZATIONS', 'NUM_EPISODES'])
                    self.AUTOSAVE_TYPE = options['AUTOSAVE_TYPE']
                else:
                    self.AUTOSAVE_TYPE = 'NUM_OPTIMIZATIONS' # Default value
                if 'AUTOSAVE_PERIOD' in keys:
                    assert(isinstance(options['AUTOSAVE_PERIOD'], int) and options['AUTOSAVE_PERIOD'] > 0)
                    self.AUTOSAVE_PERIOD = options['AUTOSAVE_PERIOD']
                else:
                    self.AUTOSAVE_PERIOD = 100000 if self.AUTOSAVE_TYPE == 'NUM_OPTIMIZATIONS' else 1000 # Default value

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
            for i in range(1, self.NUM_EPISODES[mode] + 1):
                # Clean up terminal line 
                if i % 100 == 0: print('\r                                                                                                                       ', end='')
                # Print current episode
                print(f'\r{mode}: Running episode {i} of {self.NUM_EPISODES[mode]}. Current win ratio of AGENT is {p1_score / i:.2%}.', end='')
                # Make it random who starts
                agent_start = random.choice([True, False])
                # Run one episode of the game and update running variables
                finished, turns, invalid = self.play_episode(mode=mode, agent_start=agent_start, turns=turns, invalid=invalid)
                # Update scores with winner
                if finished == 1:
                    p1_score += 1
                elif finished == 2:
                    p2_score += 1
                # Optional periodic updates
                self.perform_periodic_updates(mode=mode, episode=i)
                # Reset board to empty
                self.env.reset()
            # Current MODE Drint on new line
            print(f'\n{mode}: Average turns per episode', turns / self.NUM_EPISODES[mode])
            print(f'{mode}: Average invalid moves per episode', invalid / self.NUM_EPISODES[mode])
            print('\n')
        
            # Save final model
            if mode == 'TRAIN': self.agent.save_model()
            
    def play_episode(self, mode: str = 'TRAIN', agent: Union[None, Agent] = None, opponent: Union[None, Agent] = None, 
                     agent_start: bool = True, turns: Union[int, None] = None, invalid: Union[int, None] = None,
                     print_game: bool = False):
        """
        Let agent and opponent play one episode
        """

        # Check wether agent and opponent are given
        agent = agent if agent != None else self.agent
        opponent = opponent if opponent != None else self.opponent

        # Initialize episodic variables
        finished = -1

        # Run one game
        while finished == -1:
            ######################
            # Agent makes a turn #
            ######################
            if agent_start: # Agent starts the game
                # Get current state of the game
                state = self.env.get_state() if not type(agent) in [RandomAgent, MinimaxAgent, DeepQAgent, CQLAgent] else None
                state_var = self.env.get_state()
                # Predict best next action based on this state
                deterministic = mode in ['EVAL', 'TEST'] # Don't want randomness during evaluation
                action = agent.act(state if state else self.env, deterministic)
                # Execute action
                valid, reward, finished = self.env.step(action, self.AGENT)
                # Print board
                if print_game:
                    print('\n')
                    self.env.render_console(self.env.get_state())
                    print('AGENT action was', action if valid != -1 else 'invalid') 
                # Print and track stuff for debugging
                if self.VERBOSE:
                    if turns != None: turns += 1
                    if invalid != None and valid == -1: invalid += 1
                # Perform learning step for a learning opponent
                if self.agent.learning and mode == 'TRAIN':
                    next_state = self.env.get_state()
                    self.agent.remember(state_var, action, reward, next_state, finished)
                    self.agent.optimize_model()
                    self.num_optimizations += 1
                # End episode if somebody won or it is a tie
                if finished != -1: break
            else: # Agent doesn't start the game
                agent_start = True
            #########################
            # Opponent makes a turn #
            #########################
            # Get current state of the game
            state = self.env.get_state() if not type(opponent) in [RandomAgent, MinimaxAgent, DeepQAgent, CQLAgent] else None
            # Predict best next action based on this state
            action = opponent.act(state if state else self.env)
            # Execute action
            valid, _, finished = self.env.step(action, self.OPPONENT)
            # Print board
            if print_game:
                print('\n')
                self.env.render_console(self.env.get_state())
                print('OPPONENT action was', action if valid != -1 else 'invalid') 
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
                if getattr(self, 'UPDATE_OPPONENT', None):
                    BOOTSTRAP_EPISODES = getattr(self, 'BOOTSTRAP_EPISODES', None)
                    if episode < BOOTSTRAP_EPISODES:
                        assert(type(self.opponent) is MinimaxAgent)
                        self.opponent.decay_epsilon()
                else:
                    assert(type(self.opponent) is MinimaxAgent)
                    self.opponent.decay_epsilon() # TODO: Make decay rate and min randomness hyperparameters as well
        
        # Replace opponent for evaluation
        if mode in ['EVAL', 'TEST']:
            if getattr(self, 'REPLACE_FOR_EVALUATION', None):
                self.opponent = self.REPLACE_FOR_EVALUATION_BY
        
        # Save model periodically
        if getattr(self, 'AUTOSAVE', None):
            if self.AUTOSAVE_TYPE == 'NUM_OPTIMIZATIONS': # Periodic in number of optimizations
                if self.num_optimizations > self.AUTOSAVE_PERIOD * (self.times_saved + 1):
                    self.agent.save_model()
                    self.times_saved += 1
            else: # Episodic periodicity
                if episode > self.AUTOSAVE_PERIOD * (self.times_saved + 1):
                    self.agent.save_model()
                    self.times_saved += 1
    
    def eval(self, agent: Agent, opponent: Union[Agent, None, str] = None, episodes: int = 100, agent_start: Union[bool, None] = None, print_last_n_games: Union[int, None] = None):
        """
        Quickly evaluate the relative performance of one agent vs another with no training.

        agent_start: Union[bool, None]. If True, AGENT always starts, if False, OPPONENT always starts, if None/not specified, the starting player is chosen randomly. 
        """

        # Parse input
        if not opponent:
            opponent = RandomAgent()
        else:
            if isinstance(opponent, str):
                if opponent == 'MINIMAX':
                    opponent = MinimaxAgent(depth=3, epsilon=0.5, player=self.OPPONENT)
        if not print_last_n_games:
            print_last_n_games = 0
        else:
            assert(print_last_n_games >= 0)

        # Score counter
        p1_score = 0
        p2_score = 0
        # Keep track of total number of turns and invalid turns
        turns = 0
        invalid = 0
        
        # Play all episodes
        for i in range(1, episodes + 1):
            # Clean up terminal line 
            if i % 100 == 0: print('\r                                                                                                                       ', end='')
            # Print current episode
            print(f'\rEVAL: Running episode {i} of {episodes}. Ratios are [WINS: {p1_score / i:.2%} | LOSSES: {p2_score / i:.2%} | TIES: {(i - p1_score - p2_score) / i:.2%}]', end='')
            # Make it random who starts
            start = agent_start if agent_start != None else random.choice([True, False])
            # Run one episode of the game and update running variables
            finished, turns, invalid = self.play_episode('EVAL', agent=agent, opponent=opponent, agent_start=start, turns=turns, invalid=invalid, print_game=i>(episodes-print_last_n_games))
            # Update scores with winner
            if finished == 1:
                p1_score += 1
            elif finished == 2:
                p2_score += 1
            # Reset board to empty
            self.env.reset()
        # Done, print on new line
        print(f'\nEVAL: Average turns per episode', turns / episodes)
        print(f'EVAL: Average invalid moves per episode', invalid / episodes)
        print('\n')