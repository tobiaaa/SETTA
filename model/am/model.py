import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import ModelRegistry
from . import modules


@ModelRegistry.register
class AmplitudeMasking(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.depth = cfg.depth

        self.input_block = modules.InputBlock()

        self.res_blocks = nn.Sequential(*[
            modules.ResBlock() for i in range(self.depth)
        ])

        self.output_block = modules.OutputBlock()

        self.return_mask = False


    def forward(self, x):
        mag, phase = x[:, 0], x[:, 1]
        x = self.input_block(mag)
        x = self.res_blocks(x)
        x = self.output_block(x)
    
        x_mag = mag * x

        mag_real = x_mag * torch.cos(phase)
        mag_imag = x_mag * torch.sin(phase)

        x_out = torch.stack([mag_real, mag_imag], 1) 

        if self.return_mask:
            return x_out, x

        return x_out

    def evaluate(self, x):
        return self(x)

    def get_adapt_groups(self, groups):
        parameters = []

        for key, val in self.named_parameters():
            if 'norm' in groups and 'norm' in key:
                parameters.append(val)
            if 'output' in groups and 'output' in key:
                parameters.append(val)
            if 'input' in groups and 'input' in key:
                parameters.append(val)
            if 'time' in groups and 'time' in key:
                parameters.append(val)
            if 'local' in groups and 'local' in key:
                parameters.append(val)
            if 'freq' in groups and 'freq' in key:
                parameters.append(val)

        return parameters
