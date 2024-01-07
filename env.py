from board import ConnectFourField
import random
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import copy

FIELD_COLUMNS = 7
FIELD_ROWS = 6

"""
ASSUMPTIONS:
Player 1 is the learning player, and has ID 1
Player 2 is the Opponent and has ID 2
"""

class Env():
    """
    Implementation of the connect 4 environment.
    """
    def __init__(self, num_cols: int = FIELD_COLUMNS, num_rows: int = FIELD_ROWS):
        for v in [num_cols, num_rows]:
            if (4 > v or v > 10):
                raise ValueError("Field dimension should be between 4 and 10")
        self.field = ConnectFourField(num_cols, num_rows)
        # -1: ongoing, 0: tie, x: player x won
        self.finished = -1

    def reset(self):
        """
        Resets the field (all fields to zero)
        """
        self.field.reset()
        self.finished = -1

    def step(self, action: int, player: int) -> tuple[bool, float, int]:
        """
        Environment step method.
        Returns:
            valid: True if action was successful, False if action is illegal (column full)
            reward: Reward after playing action
            finished: -1: not finished, 0: tie, x: player x won

        Args:
            action (int): One of the columns (action is in range(FIELD_COLUMNS))
            player (int): Player_ID

        Returns:
            (bool, float, int): (valid, reward, finished)
        """
        if self.finished != -1:
            if self.finished == 0: raise ValueError(f"Game is already finished, It's a Tie!")
            else: raise ValueError(f"Game is already finished, Player {self.finished} won!")
        if action >= self.field.num_columns or action < 0:
            raise ValueError(f"Action is not valid")
        
        valid, finished = self.field.play(player, action)
        if not valid:
            # If action was invalid -> no state change, return -1 for finished, since self.finished == -1
            return False, self.compute_reward(valid, player), -1
        
        self.finished = finished
        return valid, self.compute_reward(valid, player), finished

    def compute_reward(self, valid: bool, player: int) -> float:
        """
        Computes the reward.

        Args:
            valid (bool): True (valid) or False (invalid)
            player (int): Player_ID

        Returns:
            float: Reward
        """
        if not valid: # Invalid move receives penalty
            return -0.1
        elif self.finished == player: # Player who did the move won
            return 1
        elif self.finished == 3-player or self.finished == 0: # Corresponds to Opponent. Treat a tie the same as a loss, as our goal is winning
            return -1
        else: # Try and give some reward simple for the fact that the player made a move and hasn't lost yet.
            return 0.05
        
    def get_state(self, return_type: str = 'list') -> Union[list, ConnectFourField]:
        """
        Returns the state of the environment, i.e. the game board.

        Args:
            return_type (str, optional): If 'list', returns directly the field as a list, otherwise return the ConnectFourField object. Defaults to 'list'.

        Returns:
            list or ConnectFourField: The game board.
        """
        if return_type == 'list':
            return self.field.field
        else:
            return self.field
    
    # NOTE: we had to invert the state here such that the agent that is copied over works
    # because the agent is trained to set '1's and not '2's
    def get_state_inverted(self):
        """
        Returns the inverted state of the environment, i.e. the inverted game board.

        Returns:
            list: The inverted game board
        """
        field = copy.deepcopy(self.field.field)
        field = np.array(field)
        field[field == 1] = 2
        field[field == 2] = 1
        
        return field.tolist()
    
    def random_valid_action(self) -> int:
        """
        Returns a random VALID action to perform.
        Useful for exploration or random agents.

        Returns:
            int: valid action
        """
        possible_actions = self.field.get_valid_cols()
        if possible_actions == []:
            raise ValueError("Board is full, but still attempting to play, this shouldn't happen!")
        return random.choice(possible_actions)

    def render_console(self, state: list = None):
        """
        Renders the game board on the console.

        Args:
            state (list, optional): The state of the game board. Defaults to None.
        """
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
        """
        Renders the game board beautifully.
        """
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
                    ax.add_patch(plt.Circle((j + 0.5, i + 0.5), 0.4, facecolor='yellow', edgecolor='black', linewidth=2))
                elif field_flipped[i, j] == 2:
                    # Red chip
                    ax.add_patch(plt.Circle((j + 0.5, i + 0.5), 0.4, facecolor='red', edgecolor='black', linewidth=2))

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