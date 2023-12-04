"Custom Connect-Four Environment"
import gym
from gym.spaces import Space
import torch
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, Any
from board import ConnectFourField
import random

FIELD_COLUMNS = 7
FIELD_ROWS = 6
NUM_FIELDS = FIELD_COLUMNS * FIELD_ROWS

SEED = 42

'''
ASSUMPTIONS:
Player 1 is the learning player, and has ID 1
Player 2 is the Opponent and has ID 2

Player 1 Always starts the Game
TODO: Random Selection of who can start would be better for later!
'''


class Env():
    def __init__(self, num_cols=FIELD_COLUMNS, num_rows=FIELD_ROWS, seed=SEED):
        for v in [num_cols, num_rows]:
            if (4 > v or v > 10):
                raise ValueError("Field dimension should be between 4 and 10")

        self.field = ConnectFourField(num_cols, num_rows)
        #-1: ongoing, 0: tie, x: player x won
        self.finished = -1
        self.seed = seed
        #TODO: seed the random when we need reproducibility
        #random.seed(seed)

    def reset(self):
        #Reset Field and Turn to starting player
        self.field.reset()
        self.finished = -1

    def step(self, action, player):
        #Returns:
        #If move was successful
        #Reward for this turn
        #If game is finished (and if so, who won)
        if (self.finished != -1):
            raise ValueError(f"Game is already finished, Player {self.finished} won!")
        if (action >= self.field.num_columns or action < 0):
            raise ValueError(f"Action is not valid")
        
        valid, finished = self.field.play(player, action)
        if (valid != 0):
            #move was invalid (column full)
            #TODO: give negative reward for false action
            return -1, self.compute_reward(valid, action, player), -1
        
        self.finished = finished

        return valid, self.compute_reward(valid, action, player), finished
    

    def compute_reward(self, valid, action, player):
        #TODO: give negative reward when move was invalid
        if valid == -1: # Invalid move receives penalty
            return -0.1
        elif self.finished == player: # Player who did the move won
            return 1
        else: # No reward
            return 0

    def render_console(self, state=None):
        if state is None:
            state = self.field.field
        print("_"*(2*self.field.num_columns+1))
        for row in self.field.field:
            rowString = "|"
            for el in row:
                rowString += f"{el}|"
            print(rowString)
        print("="*(2*self.field.num_columns+1))

    def get_state(self):
        return self.field.field
    
    '''
    Returns a random VALID action to perform
    Useful for Exploration or Random Agents
    '''
    def random_valid_action(self):
        if self.field.is_full():
            raise ValueError("Board is full, no action possible")
        
        possible_actions = [i for i in range(0, self.field.num_columns)]
        action = random.choice(possible_actions)

        while (self.field.is_column_full(action)):
            possible_actions.remove(action)
            action = random.choice(possible_actions)

        return action


        


