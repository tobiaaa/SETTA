import torch.nn as nn
import torch.nn.functional as F

from ..registry import ModelRegistry


@ModelRegistry.register
class BasicCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(2**i, 2**(i+1), 3, padding='same')
             for i in range(6)])
        self.output_conv = nn.Conv1d(2**6, 1, 3, padding='same')

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
            x = F.relu(x)

        x = self.output_conv(x)
        return x
