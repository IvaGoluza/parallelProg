from board import CPU, Board


class Node:
    def __init__(self, B: Board):
        self.id = id(B)
        self.B = B
        self.children = []
        self.value = None


def evaluate_node(node):
    if node.value is not None:  # this node is a leaf & it already has a value set
        return node.value

    child_values = [evaluate_node(child) for child in node.children]  # values of node children

    if node.B.LastMover == CPU and any(value == 1 for value in child_values):
        node.value = 1
    elif node.B.LastMover != CPU and any(value == -1 for value in child_values):
        node.value = -1
    elif all(value == -1 for value in child_values):
        node.value = -1
    elif all(value == 1 for value in child_values):
        node.value = 1
    else:
        node.value = sum(child_values) / len(child_values)

    return node.value
