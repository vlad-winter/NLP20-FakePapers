from LatexTokenizer import file_to_tree
from Struct import SectionNode, TextNode, WrapperNode
import re
import string


def find_wrapper_node(node, required_type):
    _required_type = required_type.lower()
    if isinstance(node, WrapperNode) and node.type.lower() == _required_type:
        return node
    if len(node.children) <= 0:
        return None
    for child in node.children:
        res = find_wrapper_node(child, required_type)
        if res is not None:
            return res


wanted_wrappernodes = ["\\textit", "\\textbf"]


def to_text(node):
    tx = []
    for child in node.children:
        if isinstance(child, TextNode):
            tx += [child.content]
        elif isinstance(child, WrapperNode) and child.type in wanted_wrappernodes:
            tx += [to_text(child)]
        elif isinstance(child, WrapperNode) and child.type == "\\cite":
            tx += ["_cite_"]
        elif isinstance(child, WrapperNode) and child.type in ["\\ref", "\\autoref"]:
            tx += ["_ref_"]
        elif isinstance(child, WrapperNode) and child.type == "\\url":
            tx += ["_url_"]

    rm_sp_from_right = ["(", "{", "[", "-"]
    rm_sp_from_left = [")", "}", "]", ",", ":", "*", "-"]

    tx = " ".join(tx)
    for sep in rm_sp_from_left:
        tx = tx.replace(" " + sep, sep)
    for sep in rm_sp_from_right:
        tx = tx.replace(sep + " ", sep)
    return tx
