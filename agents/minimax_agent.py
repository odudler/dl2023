from random import shuffle
from board import ConnectFourField
from copy import deepcopy
from .agent_interface import Agent
import random
from env import Env

class MinimaxAgent(Agent):
    """
    Implementation of the minimax agent.
    Adapted from https://github.com/AbdallahReda/Connect4/blob/master/minimaxAlphaBeta.py 
    """
    def __init__(self, env: Env, depth: int = 3, epsilon: float = 0.1, player: int = 2):
        super(MinimaxAgent, self).__init__(learning=False)
        
        self.env = env
        self.max_depth = depth
        self.player = player

        # Measure of randomness
        self.epsilon = epsilon
    
    def load_model(self, loadpath):
        pass

    def save_model(self, savepath):
        pass

    def optimize_model(self):
        pass

    def reset(self):
        self.env.reset()

    def act(self, state: list, **kwargs):
        # Choose best predicted action
        if random.random() > self.epsilon:
            return self.best_predicted_action(self.env.get_state(return_type="board"), self.max_depth, self.player)
        else:
            return self.env.random_valid_action()
        
    def decay_epsilon(self, rate: float = 0.9, min: float = 0.1):
        self.epsilon = max(min, self.epsilon * rate)
    
    def set_epsilon(self, value: float = 0.4):
        assert value >= 0.0 and value <= 1.0
        self.epsilon = value

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

    def best_predicted_action(self, board: ConnectFourField, depth: int = 4, player: int = 1):
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

