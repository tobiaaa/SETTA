import logging

import torch.nn as nn

from .registry import ModelRegistry

_logger = logging.getLogger('model')

_import_warnings = []


def replay_warnings():
    global _import_warnings
    for module, msg in _import_warnings:
        _logger.debug(f'Directory "{module}" could not be imported; Ignoring directory: {msg}')

    _import_warnings = []


@ModelRegistry.register
class Identity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, *x):
        return x[0], {}

    def evaluate(self, *x):
        return x[0]
