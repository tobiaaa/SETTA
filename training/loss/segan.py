import torch
import torch.nn as nn
import torch.nn.functional as F


class SEGANLoss(nn.Module):
    def __init__(self, cfg, mode):
        super().__init__()

    def forward(self, x_denoised, x_clean, x_noisy):
        raise NotImplementedError
