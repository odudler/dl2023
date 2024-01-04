from board import ConnectFourField
import random
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import copy
from copy import deepcopy

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
    The Opponent is integrated (Either minimax or random).
    """
    def __init__(self, opponent_type: str, minimax_depth: int = 3, minimax_epsilon: float = 0.1, num_cols: int = FIELD_COLUMNS, num_rows: int = FIELD_ROWS):
        for v in [num_cols, num_rows]:
            if (4 > v or v > 10):
                raise ValueError("Field dimension should be between 4 and 10")
        self.field = ConnectFourField(num_cols, num_rows)
        # -1: ongoing, 0: tie, x: player x won
        self.finished = -1
        
        # Either minimax or random
        self.opponent_type = opponent_type
        self.minimax_depth = minimax_depth
        self.minimax_epsilon = minimax_epsilon # Randomness parameter

    def reset(self):
        """
        Resets the field (all fields to zero)
        """
        self.field.reset()
        self.finished = -1

    def step(self, action: int) -> tuple[bool, float, int]:
        """
        Environment step method.
        Returns:
            valid: True if action was successful, False if action is illegal (column full)
            reward: Reward after playing action
            finished: -1: not finished, 0: tie, x: player x won

        Args:
            action (int): One of the columns (action is in range(FIELD_COLUMNS))

        Returns:
            (bool, float, int): (valid, reward, finished)
        """
        if self.finished != -1:
            if self.finished == 0: raise ValueError(f"Game is already finished, It's a Tie!")
            else: raise ValueError(f"Game is already finished, Player {self.finished} won!")
        if action >= self.field.num_columns or action < 0:
            raise ValueError(f"Action is not valid")
        
        # Player plays
        valid, finished = self.field.play(player=1, action=action)
        
        # If game not finished, opponent plays
        if finished == -1:
            finished = self.opponent_play()
        
        self.finished = finished
        return valid, self.compute_reward(valid), finished

    def opponent_play(self) -> int:
        opponent_action = self.get_opponent_action()
        _, finished = self.field.play(player=2, action=opponent_action)
        return finished
    
    def get_opponent_action(self) -> int:
        if self.opponent_type == 'minimax':
            if random.random() > self.minimax_epsilon:
                return self.minimax_best_predicted_action(board=self.field, depth=self.minimax_depth, player=2)
            else:
                return self.random_valid_action()
        # If not minimax, then always random actions
        else:
            return self.random_valid_action()
        
    def compute_reward(self, valid: bool) -> float:
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
        elif self.finished == 1: # Player who did the move won
            return 1
        elif self.finished == 2 or self.finished == 0: # Corresponds to Opponent. Treat a tie the same as a loss, as our goal is winning
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
    def get_state_inverted(self, return_type: str = 'list'):
        assert(return_type in ['list', 'class'])
        connect_four_field = copy.deepcopy(self.field)
        num_rows, num_columns = np.shape(connect_four_field.field)
        for i in range(num_rows):
            for j in range(num_columns):
                if connect_four_field.field[i][j] == 1:
                    connect_four_field.field[i][j] = 2
                elif connect_four_field.field[i][j] == 2:
                    connect_four_field.field[i][j] = 1
        if return_type == 'list':
            return connect_four_field.field 
        else:
            return connect_four_field
    
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
    
    """
    ######################
    # INTEGRATED MINIMAX #
    ######################
    """
    # Starting from the middle row and going outwards from there can decrease search times by a factor of over 10 
    # as the middle is in general the better column to play
    def reorder_array(self, arr):
        n = len(arr)
        middle = n // 2  # Get the index of the middle element
        reordered = [arr[middle]]  # Start with the middle element

        for i in range(1, middle+1):
            # Add the element to the right and then to the left of the middle, if they exist
            if middle + i < n:
                reordered.append(arr[middle + i])
            if middle - i >= 0:
                reordered.append(arr[middle - i])

        return reordered

    def minimax_best_predicted_action(self, board: ConnectFourField, depth: int = 4, player: int = 1):
        # Get array of possible moves
        validMoves = board.get_valid_cols()
        # Choose random starting move
        # shuffle(validMoves)
        validMoves = self.reorder_array(validMoves)

        bestMove  = validMoves[0]
        bestScore = float("-inf")

        # Initial alpha & beta values for alpha-beta pruning
        alpha = float("-inf")
        beta = float("inf")

        if player == 2: opponent = 1
        else: opponent = 2
    
        # Go through all of those moves
        for move in validMoves:
            # Create copy so as not to change the original board
            tempBoard = deepcopy(board)
            tempBoard.play(player, move)

            # Call min on that new board
            boardScore = self.minimizeBeta(tempBoard, depth - 1, alpha, beta, player, opponent)
            if boardScore > bestScore:
                bestScore = boardScore
                bestMove = move
        
        return bestMove

    def minimizeBeta(self, board: ConnectFourField, depth: int, a, b, player: int, opponent: int):
        # Get all valid moves
        validMoves = board.get_valid_cols()
        
        # RETURN CONDITION
        # Check to see if game over
        if depth == 0 or len(validMoves) == 0 or board.is_finished() != -1:
            return board.utilityValue(player)
        
        # CONTINUE TREE SEARCH
        beta = b
        # If end of tree evaluate scores
        for move in validMoves:
            boardScore = float("inf")
            # Else continue down tree as long as ab conditions met
            if a < beta:
                tempBoard = deepcopy(board)
                tempBoard.play(opponent, move)
                boardScore = self.maximizeAlpha(tempBoard, depth - 1, a, beta, player, opponent)
            if boardScore < beta:
                beta = boardScore
        return beta

    def maximizeAlpha(self, board: ConnectFourField, depth: int, a, b, player: int, opponent: int):
        # Get all valid moves
        validMoves = board.get_valid_cols()

        # RETURN CONDITION
        # Check to see if game over
        if depth == 0 or len(validMoves) == 0 or board.is_finished() != -1:
            return board.utilityValue(player)

        # CONTINUE TREE SEARCH
        alpha = a        
        # If end of tree, evaluate scores
        for move in validMoves:
            boardScore = float("-inf")
            # Else continue down tree as long as ab conditions met
            if alpha < b:
                tempBoard = deepcopy(board)
                tempBoard.play(player, move)
                boardScore = self.minimizeBeta(tempBoard, depth - 1, alpha, b, player, opponent)
            if boardScore > alpha:
                alpha = boardScore
        return alpha