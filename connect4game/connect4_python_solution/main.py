import copy
import sys
import random
import time
import queue
import numpy as np
from mpi4py import MPI
from board import Board, Evaluate, CPU, HUMAN, EMPTY  # type: ignore
from node import Node
from task import Task

DEPTH = 6
ROWS = 6
COLS = 7
LEVEL = 1


def generate_tasks(node: Node, level, iDepth, tasks):
    # return a leaf node if game end                            & set value of the node?
    if node.B.GameEnd(node.B.lastcol):
        return
    # if max level & not game end -> generate task & stop generating tree
    if level == LEVEL:
        tasks.put(Task(node, iDepth-LEVEL))
        return

    for possible_move in range(0, node.B.cols):
        if not node.B.MoveLegal(possible_move):
            continue
        B_copy = copy.deepcopy(node.B)
        B_copy.Move(possible_move, HUMAN if node.B.LastMover == CPU else CPU)
        child_node = Node(B_copy)
        node.children.append(child_node)
        generate_tasks(child_node, level+1, iDepth, tasks)


def print_tree(node, level=0):
    indent = " " * level * 2
    last_move_info = f"Last move: Column {node.B.lastcol}, Player {'CPU' if node.B.LastMover == CPU else 'HUMAN'}"
    print(f"{indent}Node({last_move_info})")
    for child in node.children:
        print_tree(child, level + 1)


def print_tasks(tasks):
    print("Generated Tasks:")
    while not tasks.empty():
        task = tasks.get()
        print("Task Depth:", task.depth)
        print("Board State:")
        for r in range(task.node.B.rows - 1, -1, -1):
            print(' '.join(map(str, task.node.B.field[r])))
        print()


def cpu_make_move(B, iDepth):
    # generate tree
    tasks = queue.Queue()
    root = Node(B)
    generate_tasks(root, 0, iDepth, tasks)
    print_tasks(tasks)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # MASTER process
    if rank == 0:
        B = Board(ROWS, COLS)
        print("Game start!")

        iDepth = DEPTH
        if len(sys.argv) > 2:
            iDepth = int(sys.argv[2])

        random.seed(time.time())

        while True:
            # Board print
            print("Current board state:")
            for r in range(ROWS - 1, -1, -1):
                print(' '.join(map(str, B.field[r])))

            # user input
            while True:
                user_input = input("Play your move! [Column 0-" + str(COLS-1) + "]:")
                if not user_input.isdigit():
                    print("Not valid! Column is a number between 0 and " + str(COLS-1))
                    continue
                user_col = int(user_input)
                if 0 <= user_col < B.Columns() and B.MoveLegal(user_col):
                    B.Move(user_col, HUMAN)
                    break
                else:
                    print("That move is not legal. Play again...")

            # game end?
            if B.GameEnd(user_col):
                print("Game end. Congrats, you are the winner!")
                for r in range(ROWS - 1, -1, -1):
                    print(' '.join(map(str, B.field[r])))
                return
            elif np.count_nonzero(B.field == EMPTY) == 0:
                print("Game end. It's a draw.")
                for r in range(ROWS - 1, -1, -1):
                    print(' '.join(map(str, B.field[r])))
                return

            # CPU's turn to play if not game end
            cpu_make_move(B, iDepth)
            dBest = -1
            iBestCol = -1
            for iCol in range(B.Columns()):
                if B.MoveLegal(iCol):
                    B.Move(iCol, CPU)
                    dResult = Evaluate(B, CPU, iCol, iDepth - 1)
                    B.UndoMove(iCol)
                    if dResult > dBest or (dResult == dBest and random.randint(0, 1) == 0):
                        dBest = dResult
                        iBestCol = iCol

            print(f"The best CPU move: {iBestCol}, value: {dBest}")
            B.Move(iBestCol, CPU)

            # game end?
            if B.GameEnd(iBestCol):
                print("Game end. CPU is the best! hehe")
                for r in range(ROWS - 1, -1, -1):
                   print(' '.join(map(str, B.field[r])))
                return
            elif np.count_nonzero(B.field == EMPTY) == 0:
                print("Game end. It's a draw.")
                for r in range(ROWS - 1, -1, -1):
                   print(' '.join(map(str, B.field[r])))
                return


if __name__ == "__main__":
    main()
