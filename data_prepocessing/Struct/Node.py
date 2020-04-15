

class Node:

    def __init__(self):
        self.children = []

    def addChild(self, child):
        self.children += [child]

    def addChildren(self, children):
        self.children += children
