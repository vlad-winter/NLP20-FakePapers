from Struct.Node import Node


class WrapperNode(Node):

    def __init__(self, type):
        Node.__init__(self)
        self.type = type
