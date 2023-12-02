"Custom Connect-Four Environment"
import gym
from gym.spaces import Space
import torch
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, Any
from board import ConnectFourField

FIELD_COLUMNS = 7
FIELD_ROWS = 6
NUM_FIELDS = FIELD_COLUMNS * FIELD_ROWS

PLAYER_ONE = 1
PLAYER_TWO = 2

STARTING_PLAYER = PLAYER_ONE

SEED = 42


class Env():
    def __init__(self, num_cols=FIELD_COLUMNS, num_rows=FIELD_ROWS, pl1=PLAYER_ONE, pl2=PLAYER_TWO, starts=STARTING_PLAYER):

        if (pl1 == 0 or pl2 == 0):
            raise(ValueError, "Player cannot have value 0")
        if (starts not in [pl1, pl2]):
            raise(ValueError, "Starting Player must be a valid player")
        for v in [num_cols, num_rows]:
            if (4 > v or v > 10):
                raise(ValueError, "Field dimension should be between 4 and 10")

        self.field = ConnectFourField(num_cols, num_rows, pl1, pl2)
        self.starting_player = starts
        self.turn = self.starting_player
        #-1: ongoing, 0: tie, x: player x won
        self.winner = -1

    def reset(self):
        #Reset Field and Turn to starting player
        self.field.reset()
        self.turn = self.starting_player
        self.winner = -1

    def step(self, action, player):
        #Returns:
        #If move was successful
        #Current board state
        #Reward for this turn
        #If game is finished (and if so, who won)

        if (self.turn != player):
            raise(ValueError, f"It is not {player}'s turn")
        if (self.winner != -1):
            raise(ValueError, f"Game is already finished, Player {self.winner} won!")
        if (action >= self.field.num_columns or action < 0):
            raise(ValueError, f"Action is not valid")
        
        valid, finished = self.field.play(player, action)
        if (valid != 0):
            #move was invalid (column full)
            #TODO: give negative reward for false action
            return -1, self.field, self.compute_reward(valid, action), -1
        
        self.winner = finished

        return valid, self.field, self.compute_reward(valid, action), finished
    

    def compute_reward(self, valid, action):
        #TODO: give negative reward when move was invalid
        return 0

    def render():
        pass

    def get_state(self):
        return self.field

