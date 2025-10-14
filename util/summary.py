import logging

import torch

logger = logging.getLogger(__name__)


def summary(model, depth=-1):
    if isinstance(model, torch.nn.Module):
        state_dict = model.state_dict()
        name = model.__class__.__name__
    elif isinstance(model, dict):
        state_dict = model
        name = 'Model'
    else:
        raise TypeError("Pass either state dict or nn.Module to summary")

    state_dict = {key: torch.numel(val) for key, val in state_dict.items()}

    root = build_tree(state_dict, name)

    print("Model Summary:")
    root.print_sub(root=True, max_depth=depth)


class Node:
    def __init__(self, name):
        self.name = name
        self.children = {}
        self.params = 0

    def print_sub(self, depth=0, indent=0, max_depth=-1, root=False):
        indent_str = '\t' * indent
        if not root:
            print(indent_str, f'{self.name}: {self.params}')
            indent += 1

        if depth == max_depth:
            return

        for child in self.children.values():
            child.print_sub(depth + 1, indent, max_depth)

        if root:
            print('---')
            print(indent_str, f'{self.name}: {self.params}')


def build_tree(state_dict, root_name):
    root = Node(root_name)

    for path, params in state_dict.items():
        node = root
        parts = path.split('.')
        for part in parts:
            if part not in node.children:
                node.children[part] = Node(part)

            node = node.children[part]
        node.params = params

    def calculate_parameters(node):
        if not node.children:
            return node.params
        else:
            total_params = sum(calculate_parameters(child) for child in node.children.values())
            node.params = total_params
            return total_params

    calculate_parameters(root)
    return root
