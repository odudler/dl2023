# Base libraries
import random
from typing import Union

# ML libraries
import torch

# Local imports
from env import Env
from agents.agent_interface import Agent
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent
#from agents.minimax_agent_old import OldMinimaxAgent
from agents.deep_q_agent import DeepQAgent
from agents.deep_q_agent_double_q import DDQAgent
from agents.cql_agent import CQLAgent
from utils import plot_rewards_steps

class Trainer():
    """
    This class is used to train and evaluate the agents.
    When initializing, the environment, the agent and the opponent should be passed.
    """
    def __init__(self, env: Union[Env, None],
                 agent: Agent,
                 opponent: Agent,
                 agent_id: int = 1,
                 opponent_id: int = 2,
                 modes: list = ['TRAIN', 'EVAL'],
                 num_episodes: Union[int, dict] = {'TRAIN': 1000, 'EVAL': 100},
                 device: torch.device = torch.device("cpu"),
                 verbose: bool = False,
                 options: Union[dict, None] = None
                 ):
        self.env = env
        self.agent = agent
        self.opponent = opponent
        # Hyperparameters
        assert all([id in [1, 2] for id in [agent_id, opponent_id]]), "Agent and Opponent IDs should be 1 or 2"
        self.AGENT = agent_id
        self.OPPONENT = opponent_id
        assert all(mode in ['TRAIN', 'EVAL'] for mode in modes), "Mode not accepted. Accepted modes: 'TRAIN' and 'EVAL'"
        self.MODES = modes
        if isinstance(num_episodes, int):
            assert(num_episodes > 0)
            self.NUM_EPISODES = {'TRAIN': num_episodes}
        else:
            assert(value > 0 for value in num_episodes.values())
            self.NUM_EPISODES = num_episodes
        assert type(device) == torch.device, "'device' should be of type 'torch.device'"
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
            if 'REPLACE_FOR_EVALUATION_BY' in keys and self.REPLACE_FOR_EVALUATION:
                assert(isinstance(options['REPLACE_FOR_EVALUATION_BY'], Agent))
                self.REPLACE_FOR_EVALUATION_BY = options['REPLACE_FOR_EVALUATION_BY']
            else:
                self.REPLACE_FOR_EVALUATION_BY = RandomAgent(env=self.env) # Default value
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
        Implements the train loop. If self.NUM_EPISODES contains an 'EVAL' key, the agent gets evaluated. 
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

            episode_rewards = []
            episode_steps = []
            
            # Play all episodes
            for i in range(1, self.NUM_EPISODES[mode] + 1):
                # Clean up terminal line 
                print('\r                                                                                                                                                       ', end='', flush=True)
                # Print current episode
                print(f'\r{mode}: Running episode {i} of {self.NUM_EPISODES[mode]}. Agent won {p1_score} times. Current win ratio of AGENT is {p1_score / i:.2%}. '
                      f'Agent Parameters: Epsilon = {self.agent.epsilon:.6f}, Memory Size = {len(self.agent.memory.memory)}', end='', flush=True)
                # Make it random who starts
                agent_start = random.choice([True, False])
                # Run one episode of the game and update running variables
                finished, turns, invalid, episode_reward, steps = self.play_episode(mode=mode, agent_start=agent_start, turns=turns, invalid=invalid)
                episode_rewards.append(episode_reward)
                episode_steps.append(steps)
                # Update scores with winner
                if finished == 1:
                    p1_score += 1
                elif finished == 2:
                    p2_score += 1
                # Optional periodic updates
                self.perform_periodic_updates(mode=mode, episode=i)
                # Reset board to empty
                self.env.reset()
            # Current MODE print on new line
            print(f'\n{mode}: Average turns per episode', turns / self.NUM_EPISODES[mode])
            print(f'{mode}: Average invalid moves per episode', invalid / self.NUM_EPISODES[mode])
            print('\n')
            plot_rewards_steps(episode_rewards, episode_steps)
        
            # Save final model
            if mode == 'TRAIN': self.agent.save_model()
            
    def play_episode(self, mode: str = 'TRAIN', agent: Union[None, Agent] = None, opponent: Union[None, Agent] = None, 
                     agent_start: bool = True, turns: Union[int, None] = None, invalid: Union[int, None] = None,
                     print_game: bool = False) -> tuple[int, Union[int, None], Union[int, None], float, int]:
        """
        Let agent and opponent play for one episode.
        Returns:
            finished: -1 if game didn't finish, otherwise Player_id to indicate who won, 0 for Tie
            turns: Total number of turns of agent
            invalid: Total number of invalid turns

        Returns:
            (int, int | None, int | None, float, int): (finished, turns, invalid, episode_reward, steps)
        """
        # Check whether agent and opponent are given
        agent = agent if agent is not None else self.agent
        opponent = opponent if opponent is not None else self.opponent

        # Initialize episodic variables
        finished = -1
        # Don't want randomness during evaluation
        deterministic = mode == 'EVAL'
        # If agent didn't start game, skip optimization after opponent play
        agent_state = None
        agent_reward = None
        episode_reward = 0
        steps = 0
        
        # Run one game
        while finished == -1:
            ######################
            # Agent makes a turn #
            ######################
            if agent_start:
                # Get current state of the game
                agent_state = self.env.get_state()
                # Predict best next action based on this state. Agent can use deterministic parameter if it has a use for it, otherwise it will be ignored
                agent_action = agent.act(agent_state, deterministic=deterministic)
                # Execute action
                agent_valid, agent_reward, agent_finished = self.env.step(agent_action, self.AGENT)
                episode_reward += agent_reward
                steps += 1
                next_state = self.env.get_state()
                # Print board
                if print_game:
                    print('\n')
                    self.env.render_console(next_state)
                    print('AGENT action was', agent_action if agent_valid else 'invalid')
                    print(f'Reward was {agent_reward}') 
                # Print and track stuff for debugging
                if self.VERBOSE:
                    turns += 1
                    if not agent_valid: invalid += 1
                # End episode if agent won or tie
                if agent_finished != -1:
                    finished = agent_finished
                    # Optimize model when game was finished by agent, otherwise we optimize after the opponent made his move (see below)
                    if self.agent.learning and mode == 'TRAIN':
                        self.agent.remember(agent_state, agent_action, agent_reward, next_state, agent_finished)
                        self.agent.optimize_model()
                        self.num_optimizations += 1
                    break
            else:
                agent_start = True
            #########################
            # Opponent makes a turn #
            #########################
            # Get current state of the game
            opponent_state = self.env.get_state()
            # Predict best next action based on this state
            opponent_action = opponent.act(state=opponent_state)
            # Execute action
            opponent_valid, opponent_reward, finished = self.env.step(opponent_action, self.OPPONENT)
            episode_reward -= opponent_reward
            steps += 1
            next_state = self.env.get_state()
            # Print board
            if print_game:
                print('\n')
                self.env.render_console(self.env.get_state())
                print('OPPONENT action was', opponent_action if opponent_valid else 'invalid')
                print(f'Reward was {agent_reward}')
            # Optimize model everytime the opponent made his move. -opponent_reward because we want to minimize opponents reward
            if agent_state and self.agent.learning and mode == 'TRAIN':
                self.agent.remember(agent_state, agent_action, -opponent_reward, next_state, finished)
                self.agent.optimize_model()
                self.num_optimizations += 1
        # Return updated variables
        return finished, turns, invalid, episode_reward, steps

    def perform_periodic_updates(self, mode: str, episode: int):
        assert(episode >= 0)
        assert(mode in ['TRAIN', 'EVAL'])

        # Update opponent periodically
        if getattr(self, 'UPDATE_OPPONENT', None):
            BOOTSTRAP_EPISODES = getattr(self, 'BOOTSTRAP_EPISODES', None)
            if (BOOTSTRAP_EPISODES and episode > BOOTSTRAP_EPISODES and episode % self.OPPONENT_UPDATE_FREQUENCY == 0) or \
            (not BOOTSTRAP_EPISODES and episode % self.OPPONENT_UPDATE_FREQUENCY == 0):
                # Update opponent with current version of the agent
                agent_class = type(self.agent)
                assert(agent_class in [CQLAgent, DeepQAgent, DDQAgent])
                self.opponent = agent_class(env=self.env, epsilon_max=0.1, epsilon_min=0.01, network_type=self.agent.network_type, device=self.DEVICE, options={'weights_init': self.agent})

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
        if mode == 'EVAL':
            if getattr(self, 'REPLACE_FOR_EVALUATION', None):
                self.opponent = self.REPLACE_FOR_EVALUATION_BY
        
        # Save model periodically
        if getattr(self, 'AUTOSAVE', None):
            if self.AUTOSAVE_TYPE == 'NUM_OPTIMIZATIONS': # Periodic in number of optimizations
                if self.num_optimizations >= self.AUTOSAVE_PERIOD * (self.times_saved + 1):
                    self.agent.save_model()
                    self.times_saved += 1
            else: # Episodic periodicity
                if episode >= self.AUTOSAVE_PERIOD * (self.times_saved + 1):
                    self.agent.save_model()
                    self.times_saved += 1
    
    def eval(self, agent: Agent, opponent: Union[Agent, None] = None, episodes: int = 100, agent_start: Union[bool, None] = None, print_last_n_games: Union[int, None] = None):
        """
        Quickly evaluate the relative performance of one agent vs another with no training.

        agent_start: Union[bool, None]. If True, AGENT always starts, if False, OPPONENT always starts, if None/not specified, the starting player is chosen randomly. 
        """

        # Parse input
        if not opponent:
            opponent = RandomAgent(env=self.env)
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
        # For plotting
        episode_rewards = []
        episode_steps = []
        
        # Play all episodes
        for i in range(1, episodes + 1):
            # Make it random who starts
            start = agent_start if agent_start is not None else random.choice([True, False])
            # Run one episode of the game and update running variables
            finished, turns, invalid, episode_reward, steps = self.play_episode('EVAL', agent=agent, opponent=opponent, agent_start=start, turns=turns, invalid=invalid, print_game=i>(episodes-print_last_n_games))
            episode_rewards.append(episode_reward)
            episode_steps.append(steps)
            # Update scores with winner
            if finished == 1:
                p1_score += 1
            elif finished == 2:
                p2_score += 1
            # Reset board to empty
            self.env.reset()
            # Clean up terminal line 
            print('\r', end='', flush=True)
            # Print current episode
            print(f'\rEVAL: Running episode {i} of {episodes}. Ratios are [WINS: {p1_score / i:.2%} | LOSSES: {p2_score / i:.2%} | TIES: {(i - p1_score - p2_score) / i:.2%}]', end='', flush=True)
        # Done, print on new line
        print(f'\nEVAL: Average turns per episode', turns / episodes)
        print(f'EVAL: Average invalid moves per episode', invalid / episodes)
        print('\n')
        plot_rewards_steps(episode_rewards, episode_steps)