import random
import time
import queue
import numpy as np
from mpi4py import MPI
from board import Board, Print_board, Evaluate, CPU, HUMAN, EMPTY
from node import Node, evaluate_node
from task import generate_tasks

DEPTH = 7
ROWS = 6
COLS = 7
LEVEL = 1  # 1: 7 tasks, 2: 49 tasks, 3: 343 tasks

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size


def send_task(task, dest):
    comm.send(task, dest=dest, tag=1)
    # print(f"Master sent task {task.node.B.LastCol} to worker {dest}", flush=True)


def terminate_workers():
    for worker in range(1, size):
        comm.send(None, dest=worker, tag=3)
        # print(f"master sent termination to worker {worker}", flush=True)


def print_tree(node, level=0):
    indent = " " * level * 2
    last_move = f"Value: {node.value}: Col {node.B.LastCol}, Player {'CPU' if node.B.LastMover == CPU else 'HUMAN'}"
    print(f"{indent}Node({last_move})", flush=True)
    for child in node.children:
        print_tree(child, level + 1)


def cpu_make_move(B, iDepth):
    # generate tree nodes & tasks
    tasks = queue.Queue()
    root = Node(B)
    nodes = {}
    generate_tasks(root, 0, iDepth, tasks, nodes, LEVEL)
    pending_results = tasks.qsize()  # number of waiting tasks results
    # print(f"0TASKS INIT: {tasks.qsize()}")
    # spread out the tasks
    free_workers = list(range(1, size))
    while not tasks.empty() or pending_results != 0:
        while free_workers and not tasks.empty():
            task = tasks.get()
            send_task(task, free_workers.pop(0))   # send task to a free worker
        # all workers have a task, check for the results
        status = MPI.Status()
        # is there any task result message available
        message_waiting = comm.Iprobe(source=MPI.ANY_SOURCE, tag=2, status=status)
        if message_waiting:
            task_result = comm.recv(source=status.Get_source(), tag=2)
            # print(f"Master accepts result from {status.Get_source()}", flush=True)
            nodes[task_result['id']].value = task_result['value']  # save the result value in its node
            pending_results = pending_results - 1
            free_workers.append(status.Get_source())  # this worker is now available
        else:
            # if there is no waiting message, master is working on a task
            if not tasks.empty():
                task = tasks.get()
                dResult = Evaluate(task.node.B, task.node.B.LastMover, task.node.B.LastCol, task.depth)
                pending_results = pending_results - 1  # one more task is done
                nodes[task.node.id].value = dResult  # save the result task number in its node using a dict
    # all tasks results are present, calculate the root result
    evaluate_node(root)
    best_root_child: Node = max(root.children, key=lambda child: child.value)  # this is the best CPU move
    B.Move(best_root_child.B.LastCol, best_root_child.B.LastMover)   # CPU makes its best move possibles
    print(f"The best CPU move: {best_root_child.B.LastCol}, value: {best_root_child.value}", flush=True)
    return B


def main():

    # MASTER process
    if rank == 0:
        B = Board(ROWS, COLS)
        print("Game start!", flush=True)
        random.seed(time.time())

        while True:
            # Board print
            Print_board(B)

            user_col = None
            # user input
            while True:
                user_input = input("Play your move! [Column 0-" + str(COLS-1) + "]:")
                if not user_input.isdigit():
                    print("Not valid! Column is a number between 0 and " + str(COLS-1), flush=True)
                    continue
                user_col = int(user_input)
                if 0 <= user_col < B.Columns() and B.MoveLegal(user_col):
                    B.Move(user_col, HUMAN)
                    break
                else:
                    print("That move is not legal. Play again...", flush=True)

            # game end?
            if B.GameEnd(user_col):
                print("Game end. Congrats, you are the winner!", flush=True)
                Print_board(B)
                terminate_workers()
                return
            elif np.count_nonzero(B.field == EMPTY) == 0:
                print("Game end. It's a draw.", flush=True)
                Print_board(B)
                terminate_workers()
                return

            # CPU's turn to play if not game end
            start_time = time.time()
            cpu_B = cpu_make_move(B, DEPTH)
            end_time = time.time()
            print(f"CPU move calculation time: {end_time - start_time:.2f} seconds", flush=True)

            # game end?
            if B.GameEnd(cpu_B.LastCol):
                print("Game end. CPU is the best! hehe", flush=True)
                Print_board(B)
                terminate_workers()
                return
            elif np.count_nonzero(B.field == EMPTY) == 0:
                print("Game end. It's a draw.", flush=True)
                Print_board(B)
                terminate_workers()
                return
    # WORKER process
    else:
        while True:
            status = MPI.Status()
            task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

            if tag == 1:
                # print(f"Worker {rank} received task {task.node.B.LastCol}.", flush=True)
                # do the task
                dResult = Evaluate(task.node.B, task.node.B.LastMover, task.node.B.LastCol, task.depth)
                # send result to the master (but include the task.node.id also)
                data = {'id': task.node.id,
                        'value': dResult}
                comm.send(data, dest=0, tag=2)
                # print(f"worker {rank} sent task result {dResult} for col {task.node.B.LastCol}.", flush=True)
            elif tag == 3:
                # there is no more tasks, master is terminating the worker process
                # print(f"Worker {rank} received termination signal.", flush=True)
                break


if __name__ == "__main__":
    main()
