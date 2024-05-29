import sys
import random
import time
from board import Board, Evaluate, CPU, HUMAN # type: ignore

DEPTH = 6

def main():
    if len(sys.argv) < 2:
        print("Uporaba: <program> <fajl s trenutnim stanjem> [<dubina>]")
        return

    B = Board()
    B.Load(sys.argv[1])
    iDepth = DEPTH
    if len(sys.argv) > 2:
        iDepth = int(sys.argv[2])

    random.seed(time.time())

    for iCol in range(B.Columns()):
        if B.GameEnd(iCol):
            print("Igra zavrsena!")
            return

    while True:
        print(f"Dubina: {iDepth}")
        dBest = -1
        iBestCol = -1
        for iCol in range(B.Columns()):
            if B.MoveLegal(iCol):
                if iBestCol == -1:
                    iBestCol = iCol
                B.Move(iCol, CPU)
                dResult = Evaluate(B, CPU, iCol, iDepth - 1)
                B.UndoMove(iCol)
                if dResult > dBest or (dResult == dBest and random.randint(0, 1) == 0):
                    dBest = dResult
                    iBestCol = iCol
                print(f"Stupac {iCol}, vrijednost: {dResult}")

        iDepth //= 2
        if dBest != -1 or iDepth <= 0:
            break

    print(f"Najbolji: {iBestCol}, vrijednost: {dBest}")
    B.Move(iBestCol, CPU)
    B.Save(sys.argv[1])

    for iCol in range(B.Columns()):
        if B.GameEnd(iCol):
            print("Igra zavrsena! (pobjeda racunala)")
            return

if __name__ == "__main__":
    main()
