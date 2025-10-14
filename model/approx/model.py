import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import ModelRegistry
from .map import LinearMap
from .vector import VectorMap
from .affine import AffineMap
from .nn import NNMap


@ModelRegistry.register
class Approximator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        if self.cfg.method == 'map':
            self.method = LinearMap()
        elif self.cfg.method == 'vector':
            self.method = VectorMap()
        elif self.cfg.method == 'affine':
            self.method = AffineMap()
        elif self.cfg.method == 'nn':
            self.method = NNMap()
        else:
            raise ValueError(f'Approximation method "{self.cfg.method}" unknown')

    def forward(self, *x):
        return self.method(*x)

    def evaluate(self, *x):
        return self.method.evaluate(*x)

    def finalize(self):
        self.method.finalize()

    def save_embs(self, path):
        self.method.save_embs(path)

    def save(self, path):
        self.method.save(path)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self.method.load_state_dict(state_dict, strict, assign)
