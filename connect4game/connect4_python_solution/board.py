import numpy as np

EMPTY = 0
CPU = 1
HUMAN = 2

class Board:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.LastMover = EMPTY
        self.lastcol = -1
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
        self.lastcol = col
        return True

    def UndoMove(self, col):
        if self.height[col] == 0:
            return False
        self.height[col] -= 1
        self.field[self.height[col], col] = EMPTY
        return True

    def GameEnd(self, lastcol):
        row = self.height[lastcol] - 1
        if row < 0:
            return False
        player = self.field[row, lastcol]

        # Check vertical
        if row >= 3 and np.all(self.field[row-3:row+1, lastcol] == player):
            return True

        # Check horizontal
        for c in range(max(0, lastcol-3), min(self.cols-3, lastcol+1)):
            if np.all(self.field[row, c:c+4] == player):
                return True

        # Check diagonal (/)
        for dr, dc in [(-1, 1), (1, -1)]:
            seq = 0
            for k in range(-3, 4):
                r, c = row + dr * k, lastcol + dc * k
                if 0 <= r < self.rows and 0 <= c < self.cols and self.field[r, c] == player:
                    seq += 1
                    if seq == 4:
                        return True
                else:
                    seq = 0

        return False

    def Load(self, fname):
        with open(fname, 'r') as f:
            self.rows, self.cols = map(int, f.readline().split())
            self.field = np.zeros((self.rows, self.cols), dtype=int)
            for r in range(self.rows-1, -1, -1):
                self.field[r] = list(map(int, f.readline().split()))
        self.height = np.sum(self.field != EMPTY, axis=0)

    def Save(self, fname):
        with open(fname, 'w') as f:
            f.write(f"{self.rows} {self.cols}\n")
            for r in range(self.rows-1, -1, -1):
                f.write(' '.join(map(str, self.field[r])) + '\n')

def Evaluate(Current, LastMover, iLastCol, iDepth):
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
            Current.Move(iCol, NewMover)
            dResult = Evaluate(Current, NewMover, iCol, iDepth - 1)
            Current.UndoMove(iCol)
            if dResult > -1:
                bAllLose = False
            if dResult != 1:
                bAllWin = False
            if dResult == 1 and NewMover == CPU:
                return 1
            if dResult == -1 and NewMover == HUMAN:
                return -1
            dTotal += dResult

    if bAllWin:
        return 1
    if bAllLose:
        return -1

    return dTotal / iMoves
