from Struct.Node import Node


class TextNode(Node):

    def __init__(self, content):
        Node.__init__(self)
        self.content = content
