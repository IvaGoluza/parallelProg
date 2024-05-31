import sys

import numpy as np

EMPTY = 0
CPU = 1
HUMAN = 2


class Board:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.LastMover = EMPTY
        self.LastCol = -1
        self.field = np.full((rows, cols), EMPTY)
        self.height = np.zeros(cols, dtype=int)

    def Columns(self):
        return self.cols

    def MoveLegal(self, col):
        return self.field[self.rows - 1, col] == EMPTY

    def Move(self, col, player):
        if not self.MoveLegal(col):
            return False
        self.field[self.height[col], col] = player
        self.height[col] += 1
        self.LastMover = player
        self.LastCol = col
        return True

    def UndoMove(self, col, prevCol, prevMover):
        if self.height[col] == 0:
            return False
        self.height[col] -= 1
        self.field[self.height[col], col] = EMPTY
        self.LastCol = prevCol
        self.LastMover = prevMover
        return True

    def GameEnd(self, last_col):
        row = self.height[last_col] - 1
        if row < 0:
            return False
        player = self.field[row, last_col]

        # Check vertical
        if row >= 3 and np.all(self.field[row-3:row+1, last_col] == player):
            return True

        # Check horizontal
        for c in range(max(0, last_col - 3), min(self.cols - 3, last_col + 1)):
            if np.all(self.field[row, c:c+4] == player):
                return True

            # Check diagonal (/)
            for dr, dc in [(-1, 1), (1, -1)]:
                seq = 0
                for k in range(-3, 4):
                    r, c = row + dr * k, last_col + dc * k
                    if 0 <= r < self.rows and 0 <= c < self.cols and self.field[r, c] == player:
                        seq += 1
                        if seq == 4:
                            return True
                    else:
                        seq = 0

            # Check diagonal (\)
            for dr, dc in [(1, 1), (-1, -1)]:
                seq = 0
                for k in range(-3, 4):
                    r, c = row + dr * k, last_col + dc * k
                    if 0 <= r < self.rows and 0 <= c < self.cols and self.field[r, c] == player:
                        seq += 1
                        if seq == 4:
                            return True
                    else:
                        seq = 0
        return False


def Print_board(B: Board):
    print("Current board state:", flush=True)
    print('*' + '-' * (2 * B.cols - 1) + '*', flush=True)

    for r in range(B.rows - 1, -1, -1):
        row = []
        for cell in B.field[r]:
            if cell == 2:
                row.append('X')
            elif cell == 1:
                row.append('O')
            else:
                row.append(' ')
        print('|' + '|'.join(row) + '|', flush=True)

    print('*' + '-' * (2 * B.cols - 1) + '*', flush=True)
    print(' ' + ' '.join(map(str, range(B.cols))), flush=True)
    sys.stdout.flush()


def Evaluate(Current: Board, LastMover, iLastCol, iDepth):
    if Current.GameEnd(iLastCol):
        return 1 if LastMover == CPU else -1

    if iDepth == 0:
        return 0

    NewMover = HUMAN if LastMover == CPU else CPU
    dTotal = 0
    iMoves = 0
    bAllLose = True
    bAllWin = True

    for iCol in range(Current.Columns()):
        if Current.MoveLegal(iCol):
            iMoves += 1
            prevCol = Current.LastCol
            prevMover = Current.LastMover
            Current.Move(iCol, NewMover)
            dResult = Evaluate(Current, NewMover, iCol, iDepth - 1)
            Current.UndoMove(iCol, prevCol, prevMover)
            if dResult > -1:
                bAllLose = False
            if dResult != 1:
                bAllWin = False
            if dResult == 1 and prevMover == CPU:
                return 1
            if dResult == -1 and prevMover == HUMAN:
                return -1
            dTotal += dResult

    if bAllWin:
        return 1
    if bAllLose:
        return -1

    return dTotal / iMoves
