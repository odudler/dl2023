"Implementation of Connect-Four Field"

class ConnectFourField():

    def __init__(self, num_columns, num_rows, player_one, player_two):
        self.field = [[0 for _ in range(num_rows)] for _ in range(num_columns)]
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.player_one_val = player_one
        self.player_two_val = player_two

    def reset(self):
        self.field = [[0 for _ in range(self.num_rows)] for _ in range(self.num_columns)]

    def is_column_full(self, col):
        #Test if the entry in the col of the highest row is non-zero
        return self.field[0][col] != 0
    
    def get_col_free_entry(self, col):
        #returns the row which is the lowest non-occupied row in that column
        #returns -1 if the col is full
        for i in range(self.num_rows, -1, -1):
            if self.field[i][col] == 0:
                return i
            
        return -1
    
    def is_finished(self):
        #Either the board is full, or one player has 4 connected "dots"
        #returns -1 if not finished, 0 if tie, or player_id if player won
        if self.four_connected() != 0:
            return self.four_connected()
        elif self.is_full():
            return 0
        else:
            return -1

    def is_full(self):
        #Return if all entries in the field are filled out (non-zero)
        for row in self.field:
            for el in row:
                if el == 0:
                    return False
        return True

    
    def four_connected(self):
        #Returns: 
        # 0: no 4 connected
        # 1: Player 1 connected 4
        # 2: Player 2 connected 4
        return self.connected_val(self.player_one_val, 4) or self.four_connected_value(self.player_two_val, 4)

    def connected_val(self,val, streak):
        #Check if a certain val appears streak amount of times in a row
        #column or diagonally
        if streak > min(self.num_columns, self.num_rows):
            return False

        # Check horizontally
        for row in self.field:
            for i in range(len(row) - (streak-1)):
                if all(row[i + j] == val for j in range(streak)):
                    return True

        # Check vertically
        for col in range(len(self.field[0])):
            for i in range(len(self.field) - (streak-1)):
                if all(self.field[i + j][col] == val for j in range(streak)):
                    return True

        # Check diagonally (from top-left to bottom-right)
        for i in range(len(self.field) - (streak-1)):
            for j in range(len(self.field[0]) - (streak-1)):
                if all(self.field[i + k][j + k] == val for k in range(streak)):
                    return True

        # Check diagonally (from top-right to bottom-left)
        for i in range(len(self.field) - (streak-1)):
            for j in range((streak-1), len(self.field[0])):
                if all(self.field[i + k][j - k] == val for k in range(streak)):
                    return True

        return False
    
    def play(self, player, action):
        #returns tuple (successful, finished)
        #successful: 0 if move was successful, -1 if the given action is illegal (column full)
        #finished: -1: not finished, 0: tie, x: player x won
        row = self.get_col_free_entry(action)
        if row == -1:
            return -1, 0
        
        self.field[row][action] = player

        return 0, self.is_finished()

            
