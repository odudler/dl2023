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

    def connected_val(self, val, streak):
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
        """
        returns tuple (successful, finished)
        successful: 0 if move was successful, -1 if the given action is illegal (column full)
        finished: -1: not finished, 0: tie, x: player x won
        """
        
        row = self.get_col_free_entry(action)
        if row == -1:
            return -1, 0
        
        self.field[row][action] = player

        return 0, self.is_finished()

    # Adapted from https://github.com/AbdallahReda/Connect4/blob/master/utility.py#L6
    def countSequence(self, player: int, length: int):
        """ Given the board state , the current player and the length of Sequence you want to count
            Return the count of Sequences that have the give length
        """
        def verticalSeq(row, col):
            """Return 1 if it found a vertical sequence with the required length 
            """
            count = 0
            for rowIndex in range(row, self.num_rows):
                if self.field[rowIndex][col] == self.field[row][col]:
                    count += 1
                else:
                    break
            if count >= length:
                return 1
            else:
                return 0

        def horizontalSeq(row, col):
            """Return 1 if it found a horizontal sequence with the required length 
            """
            count = 0
            for colIndex in range(col, self.num_columns):
                if self.field[row][colIndex] == self.field[row][col]:
                    count += 1
                else:
                    break
            if count >= length:
                return 1
            else:
                return 0

        def negDiagonalSeq(row, col):
            """Return 1 if it found a negative diagonal sequence with the required length 
            """
            count = 0
            colIndex = col
            for rowIndex in range(row, -1, -1):
                if colIndex > self.num_rows:
                    break
                elif self.field[rowIndex][colIndex] == self.field[row][col]:
                    count += 1
                else:
                    break
                colIndex += 1 # increment column when row is incremented
            if count >= length:
                return 1
            else:
                return 0

        def posDiagonalSeq(row, col):
            """Return 1 if it found a positive diagonal sequence with the required length 
            """
            count = 0
            colIndex = col
            for rowIndex in range(row, self.num_rows):
                if colIndex > self.num_rows:
                    break
                elif self.field[rowIndex][colIndex] == self.field[row][col]:
                    count += 1
                else:
                    break
                colIndex += 1 # increment column when row incremented
            if count >= length:
                return 1
            else:
                return 0

        totalCount = 0
        # for each piece in the board...
        for row in range(self.num_rows):
            for col in range(self.num_columns):
                # ...that is of the player we're looking for...
                if self.field[row][col] == player:
                    # check if a vertical streak starts at (row, col)
                    totalCount += verticalSeq(row, col)
                    # check if a horizontal four-in-a-row starts at (row, col)
                    totalCount += horizontalSeq(row, col)
                    # check if a diagonal (both +ve and -ve slopes) four-in-a-row starts at (row, col)
                    totalCount += (posDiagonalSeq(row, col) + negDiagonalSeq(row, col))
        # return the sum of sequences of length 'length'
        return totalCount

    def utilityValue(self, player: int):
        """ A utility fucntion to evaluate the state of the board and report it to the calling function,
            utility value is defined as the score of the player who calles the function - score of opponent player,
            The score of any player is the sum of each sequence found for this player scaled by a large factor for
            sequences with higher lengths.
        """

        if player == 1: opponent = 2
        else: opponent = 1

        p4s    = self.countSequence(player, 4)
        p3s   = self.countSequence(player, 3)
        p2s     = self.countSequence(player, 2)
        playerScore    = p4s * 99999 + p3s * 999 + p2s * 99

        o4s  = self.countSequence(opponent, 4)
        o3s = self.countSequence(opponent, 3)
        o2s   = self.countSequence(opponent, 2)
        opponentScore  = o4s * 99999 + o3s * 999 + o2s * 99

        if o4s > 0:
            # Current player lost the game
            # Return biggest negative value
            return float('-inf')
        else:
            # Return difference in scores
            return playerScore - opponentScore

