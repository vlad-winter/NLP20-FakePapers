import re
from Struct import WrapperNode, TextNode, SectionNode


not_to_parse_types = ['equation', 'figure', 'split', 'align', 'table', 'tabular', 'minipage']


def is_token_seq(tokens, i, seq):
    for j in range(len(seq)):
        if tokens[i+j] != seq[j]:
            return False
    return True


def build_tree(tokens):
    n = len(tokens)
    i = 0
    children = []
    while i < n:
        if tokens[i] == '\\begin':
            i += 1
            if tokens[i] != "{":
                raise ValueError
            i += 1
            type = []
            while tokens[i] != '}':
                type += [tokens[i]]
                i += 1
            i += 1
            node = WrapperNode(" ".join(type))
            seq = ["{"] + type + ["}"]
            sub_tokens = []
            while not(tokens[i] == '\\end' and is_token_seq(tokens, i+1, seq)):
                sub_tokens += [tokens[i]]
                i += 1
            i += len(seq) + 1
            if type[0] not in not_to_parse_types:
                node_children = build_tree(sub_tokens)
                node.addChildren(node_children)
            children += [node]
        elif tokens[i] == '\\section':
            i += 1
            if tokens[i] == "*":
                i += 1
            if tokens[i] != "{":
                raise ValueError
            i += 1
            title = []
            while tokens[i] != '}':
                title += [tokens[i]]
                i += 1
            i += 1
            node = SectionNode(" ".join(title))
            sub_tokens = []
            while i < n and tokens[i] != '\\section':
                sub_tokens += [tokens[i]]
                i += 1
            node_children = build_tree(sub_tokens)
            node.addChildren(node_children)
            children += [node]
        elif tokens[i][0] == '\\':
            curr_token = tokens[i]
            i += 1
            if i < n:
                if tokens[i] == '[':
                    i += 1
                    specs = []
                    while tokens[i] != ']':
                        specs += [tokens[i]]
                        i += 1
                    i += 1
                if tokens[i] == '{':
                    opened_brackets = 1
                    node = WrapperNode(curr_token)
                    i += 1
                    sub_tokens = []
                    while opened_brackets > 0:
                        if tokens[i] == '{':
                            opened_brackets += 1
                        elif tokens[i] == '}':
                            opened_brackets -= 1
                        if opened_brackets > 0:
                            sub_tokens += [tokens[i]]
                            i += 1
                    i += 1
                    node_children = build_tree(sub_tokens)
                    node.addChildren(node_children)

                    if i < n and tokens[i] == '{':  # a two-bracket command
                        opened_brackets = 1
                        i += 1
                        sub_tokens = []
                        while opened_brackets > 0:
                            if tokens[i] == '{':
                                opened_brackets += 1
                            elif tokens[i] == '}':
                                opened_brackets -= 1
                            if opened_brackets > 0:
                                sub_tokens += [tokens[i]]
                                i += 1
                        i += 1
                        node_children = build_tree(sub_tokens)
                        node.addChildren(node_children)

                    children += [node]
                else:
                    children += [TextNode(curr_token)]
        elif tokens[i] == '$':
            x = i
            i += 1
            double = False
            if tokens[i] == '$':
                double = True
                i += 1
            eq = []
            while tokens[i] != '$':
                eq += [tokens[i]]
                i += 1
            i += 1
            if double:
                if tokens[i] != '$':
                    raise ValueError
                i += 1

            if double:
                children += [TextNode(f'_eq')]
            else:
                children += [TextNode(f'_inline_eq_')]
        else:
            children += [TextNode(tokens[i])]
            i += 1
    return children


def tokenize(s):
    s = ' '.join(s).replace('\n', ' ').replace("\\", " \\").strip()
    s = parse_sentence(s).strip()
    tokens = s.split(" ")
    tree = WrapperNode('Tree')
    tree.addChildren(build_tree(tokens))
    return tree


def parse_sentence(s):
    seperators = ["(", ")", "=", "+", "-", "*", ":", "{", "}", "$", "[", "]", ","]
    num_free = re.sub("\d+", "N", s)
    new_text = ""
    for w in num_free.split(" "):
        new_w = ""
        for x in w:
            if x in seperators:
                x = " " + x + " "
            new_w += x
        new_text += (new_w + " ")
    s_with_points_floats = re.sub(' +', ' ', new_text)
    new_s = ""
    for w2 in s_with_points_floats.split(" "):
        if 'N.N' in w2:
            w2 = 'N'
        new_s += (w2 + " ")
    return new_s[:-1]


def file_to_tree(file_path):
    f = open(file_path)

    lines = f.readlines()
    lines = filter(lambda a: a != '\n', lines)

    all = []

    for l in lines:
        l = re.split(r'%(?<!\\%)', l)[0]
        l = l.replace("\\\\", "\n")
        all += [l.strip()]

    t = tokenize(all)

    return t
