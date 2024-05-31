import copy

from board import CPU, HUMAN
from node import Node


class Task:
    def __init__(self, node, depth):
        self.node = node
        self.depth = depth


def generate_tasks(node: Node, level, iDepth, tasks, nodes, agglomeration_level):
    # add a node in nodes map (map value setter)
    nodes[node.id] = node
    # set node value if game end
    if node.B.GameEnd(node.B.LastCol):
        node.value = 1 if node.B.LastMover == CPU else -1
        return
    # if max level & not game end -> generate task & stop generating tree
    if level == agglomeration_level:
        task = Task(node, iDepth-level)
        tasks.put(task)
        return

    for possible_move in range(0, node.B.cols):
        if not node.B.MoveLegal(possible_move):
            continue
        B_copy = copy.deepcopy(node.B)
        B_copy.Move(possible_move, HUMAN if node.B.LastMover == CPU else CPU)
        child_node = Node(B_copy)
        node.children.append(child_node)
        generate_tasks(child_node, level+1, iDepth, tasks, nodes, agglomeration_level)
