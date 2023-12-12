"Implementation of Connect-Four Field"

class ConnectFourField():

    def __init__(self, num_columns, num_rows):
        self.field = [[0 for _ in range(num_columns)] for _ in range(num_rows)]
        self.num_columns = num_columns
        self.num_rows = num_rows

    def reset(self):
        self.field = [[0 for _ in range(self.num_columns)] for _ in range(self.num_rows)]

    def is_column_full(self, col):
        #Test if the entry in the col of the highest row is non-zero
        return self.field[0][col] != 0
    
    def get_col_free_entry(self, col):
        #returns the row which is the lowest non-occupied row in that column
        #returns -1 if the col is full
        for i in range(self.num_rows-1, -1, -1):
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
        
    def get_valid_cols(self):
        valid_locations = []
        for i in range(0,self.num_columns):
            if not self.is_column_full(i):
                valid_locations.append(i)
        return valid_locations

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
        if self.connected_val(1, 4) != []:
            return 1
        elif self.connected_val(2, 4) != []:
            return 2
        else:
            return 0

    def connected_val(self,val, streak):
        #Check if a certain val appears streak amount of times in a row
        #column or diagonally
        #Returns: the first and last entry of the streak as array [row_1, col_1, row_2, col_2]
        if streak > max(self.num_columns, self.num_rows):
            return False

        # Check horizontally
        for index, row in enumerate(self.field):
            for i in range(len(row) - (streak-1)):
                if all(row[i + j] == val for j in range(streak)):
                    return [index, i, index, i+streak-1]
                    #return True

        # Check vertically
        for col in range(len(self.field[0])):
            for i in range(len(self.field) - (streak-1)):
                if all(self.field[i + j][col] == val for j in range(streak)):
                    return [i, col, i+streak-1, col]

        # Check diagonally (from top-left to bottom-right)
        for i in range(len(self.field) - (streak-1)):
            for j in range(len(self.field[0]) - (streak-1)):
                if all(self.field[i + k][j + k] == val for k in range(streak)):
                    return [i, j, i+streak-1, j+streak-1]
                    #return True

        # Check diagonally (from top-right to bottom-left)
        for i in range(len(self.field) - (streak-1)):
            for j in range((streak-1), len(self.field[0])):
                if all(self.field[i + k][j - k] == val for k in range(streak)):
                    return [i, j, i+streak-1, j-streak+1]
                    #return True

        return []
    
    def play(self, player, action):
        #returns tuple (successful, finished)
        #successful: 0 if move was successful, -1 if the given action is illegal (column full)
        #finished: -1: not finished, 0: tie, x: player x won
        row = self.get_col_free_entry(action)
        if row == -1:
            return -1, 0
        
        self.field[row][action] = player

        return 0, self.is_finished()

            
