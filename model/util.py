import torch.nn as nn

from .registry import ModelRegistry


@ModelRegistry.register
class Identity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, *x):
        return x[0], {}

    def evaluate(self, *x):
        return x[0]
