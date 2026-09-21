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

    output = "Model Summary:"
    output += root.print_sub(root=True, max_depth=depth)

    summary_logger = logging.getLogger('summary')
    summary_logger.info(output)


class Node:
    def __init__(self, name):
        self.name = name
        self.children = {}
        self.params = 0

    def print_sub(self, depth=0, indent=0, max_depth=-1, root=False):
        indent_str = '\t' * indent
        output = ""
        if not root:
            output += '\n' + indent_str
            output += f'{self.name}: {self.params:,}'
            indent += 1

        if depth == max_depth:
            return output

        for child in self.children.values():
            output += child.print_sub(depth + 1, indent, max_depth)

        if root:
            output += '\n---\n'
            output += indent_str
            output += f'{self.name}: {self.params:,}'

        return output


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
