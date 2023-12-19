"Implementation of Agent using minimax to play"
from .agent_interface import Agent
from board import ConnectFourField
import typing
import math
import random
import copy
from env import Env

# NOTE: This Agent only works for the fixed board size of 7x6!!!!!!
# NOTE: This Agent-Implementation was taken from: https://github.com/bryanjsanchez/ConnectFour/blob/master/connectfour.py

EMPTY = 0
MAX_SPACE_TO_WIN = 3

class OldMinimaxAgent(Agent):
    def __init__(self, env, epsilon=0.3, whoami=1): #Default "whoami" is being the AGENT
        self.env = env
        self.epsilon = epsilon
        #This variable is necessary as the minimax algo needs to know if its the "1" (AGENT) or the "2" (OPPONENT)
        self.whoami = whoami

    def load_model(self, loadpath):
        pass

    def save_model(self, savepath):
        pass

    def optimize_model(self):
        pass

    def reset(self):
        pass

    def act(self, state):
        if random.random() > self.epsilon:
            board = copy.deepcopy(state)
            action, _ = _minimax(board, 3, -math.inf, math.inf, True, self.whoami)
            return action
        else:
            return self.env.random_valid_action()

# Need to clone the board in order to "preplay" the game without altering the actual game
def _clone_and_place_piece(board: ConnectFourField, player, column): #TODO: Player should be whoever is currently making a move!
    new_board = copy.deepcopy(board)
    new_board.play(player, column)
    return new_board

# Calculate the scores for possible moves to find the best one
def _score(board: ConnectFourField, player):
    score = 0
    # Give more weight to center columns
    for col in range(2, 5):
        for row in range(board.num_rows):
            if board.field[row][col] == player:
                if col == 3:
                    score += 3
                else:
                    score+= 2
    # Horizontal pieces
    for col in range(board.num_columns - MAX_SPACE_TO_WIN):
        for row in range(board.num_rows):
            adjacent_pieces = [board.field[row][col], board.field[row][col+1], 
                                board.field[row][col+2], board.field[row][col+3]] 
            score += _evaluate_adjacents(adjacent_pieces, player)
    # Vertical pieces
    for col in range(board.num_columns):
        for row in range(board.num_rows - MAX_SPACE_TO_WIN):
            adjacent_pieces = [board.field[row][col], board.field[row+1][col], 
                                board.field[row+2][col], board.field[row+3][col]] 
            score += _evaluate_adjacents(adjacent_pieces, player)
    # Diagonal upwards pieces
    for col in range(board.num_columns - MAX_SPACE_TO_WIN):
        for row in range(board.num_rows - MAX_SPACE_TO_WIN):
            adjacent_pieces = [board.field[row][col], board.field[row+1][col+1], 
                                board.field[row+2][col+2], board.field[row+3][col+3]] 
            score += _evaluate_adjacents(adjacent_pieces, player)
    # Diagonal downwards pieces
    for col in range(board.num_columns - MAX_SPACE_TO_WIN):
        for row in range(MAX_SPACE_TO_WIN, board.num_rows):
            adjacent_pieces = [board.field[row][col], board.field[row-1][col+1], 
                    board.field[row-2][col+2], board.field[row-3][col+3]]
            score += _evaluate_adjacents(adjacent_pieces, player)
    return score

# Evaluate scores for considering the adjacent pieces
def _evaluate_adjacents(adjacent_pieces, player):
    opponent = 3-player
    score = 0
    player_pieces = 0
    empty_spaces = 0
    opponent_pieces = 0
    for p in adjacent_pieces:
        if p == player:
            player_pieces += 1
        elif p == EMPTY:
            empty_spaces += 1
        elif p == opponent:
            opponent_pieces += 1
    if player_pieces == 4:
        score += 99999
    elif player_pieces == 3 and empty_spaces == 1:
        score += 100
    elif player_pieces == 2 and empty_spaces == 2:
        score += 10
    return score

# board: copy of the current playing field, ply: depth of the tree, alpha/beta: limits of the score, for better performance, maxi_player: not sure what this is yet
def _minimax(board : ConnectFourField, ply, alpha, beta, maxi_player, whoami):
    valid_cols = board.get_valid_cols()
    finished = board.is_finished()
    if ply == 0 or finished != -1:
        if finished != -1:
            if finished == 3-whoami:
                return (None,-1000000000)
            elif finished == whoami:
                return (None,1000000000)
            else: # There is no winner
                return (None,0)
        else: # Ply == 0
            return (None,_score(board, whoami)) # TODO: 
    # If max player
    if maxi_player:
        value = -math.inf
        # If every choice has an equal score, choose randomly
        col = random.choice(valid_cols)
        # Expand current node/board
        for c in valid_cols:
            next_board = _clone_and_place_piece(board, 3-whoami, c)
            new_score = _minimax(next_board, ply - 1, alpha, beta, False, whoami)[1]
            if new_score > value:
                value = new_score
                col = c
            # Alpha pruning
            if value > alpha:
                alpha = new_score
            # If beta is less than or equal to alpha, there will be no need to
            # check other branches because there will not be a better move
            if beta <= alpha:
                break
        return col, value
    #if min player
    else:
        value = math.inf
        col = random.choice(valid_cols)
        for c in valid_cols:
            next_board = _clone_and_place_piece(board, whoami, c)
            new_score = _minimax(next_board, ply - 1, alpha, beta, True, whoami)[1]
            if new_score < value:
                value = new_score
                col = c
            if value < beta:
                beta  = value
            if beta <= alpha:
                break
        return col, value
