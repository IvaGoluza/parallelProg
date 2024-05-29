from board import EMPTY, Board


class Node:
    def __init__(self, B: Board):
        self.B = B
        self.children = []
        self.value = EMPTY


