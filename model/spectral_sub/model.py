import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import ModelRegistry


@ModelRegistry.register
class SpectralSubtraction(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_avg = cfg.n_avg

    @torch.no_grad()
    def forward(self, x):
        x_mag = x[:, 0]
        phase = x[:, 1]

        x_noise, _ = torch.topk(x_mag ** 2, self.n_avg, -1, largest=False)
        x_noise = x_noise.mean(-1)[..., None]

        x_mag = x_mag - x_noise
        x_mag = F.relu(x_mag) 

        mag_real = x_mag * torch.cos(phase)
        mag_imag = x_mag * torch.sin(phase)

        x_out = torch.stack([mag_real, mag_imag], 1) 

        return x_out
