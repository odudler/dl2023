"Custom Connect-Four Environment"
import gym
from gym.spaces import Space
import torch
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, Any
from board import ConnectFourField
import random
import matplotlib.pyplot as plt
import numpy as np
import copy

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
        
    def get_state(self):
        return self.field.field
    
    def get_state_inverted(self):
        field = copy.deepcopy(self.field.field)
        num_rows, num_columns = np.shape(field)
        for i in range(num_rows):
            for j in range(num_columns):
                if field[i][j] == 1:
                    field[i][j] = 2
                elif field[i][j] == 2:
                    field[i][j] = 1
        return field

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

    

    def render_pretty(self):
        # Flip the board horizontally
        field_flipped = np.flipud(self.field.field)

        num_rows, num_columns = np.shape(field_flipped)

        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(num_columns, num_rows))
        
        # Set background color
        ax.set_facecolor('lightblue')

        # Plot gridlines
        for i in range(num_rows + 1):
            ax.axhline(i, color='white', linewidth=2)

        for j in range(num_columns + 1):
            ax.axvline(j, color='white', linewidth=2)

        # Plot chips
        for i in range(num_rows):
            for j in range(num_columns):
                if field_flipped[i, j] == 1:
                    # Yellow chip
                    ax.add_patch(plt.Circle((j + 0.5, i + 0.5), 0.4, color='yellow', edgecolor='black', linewidth=2))
                elif field_flipped[i, j] == 2:
                    # Red chip
                    ax.add_patch(plt.Circle((j + 0.5, i + 0.5), 0.4, color='red', edgecolor='black', linewidth=2))

        # Check if four are connected, if so, draw a line through them
        #NOTE: need to flip the row value as the plot starts from bottom but we start from top (row 0 is highest row...)
        streak_yellow = self.field.connected_val(1, 4)
        streak_red = self.field.connected_val(2,4)

        if (streak_yellow != []):
            ax.plot([streak_yellow[1] + 0.5, streak_yellow[3] + 0.5], [5 - streak_yellow[0] + 0.5, 5 - streak_yellow[2] + 0.5], color='black', linewidth=7)
        if (streak_red != []):
            ax.plot([streak_red[1] + 0.5, streak_red[3] + 0.5], [5 - streak_red[0] + 0.5, 5 - streak_red[2] + 0.5], color='black', linewidth=7)

        # Set axis limits and remove ticks
        ax.set_xlim(0, num_columns)
        ax.set_ylim(0, num_rows)
        ax.set_xticks([])
        ax.set_yticks([])

        # Display the plot
        plt.show()

    
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


        


