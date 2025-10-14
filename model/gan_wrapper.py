import torch
import torch.nn as nn
import torch.nn.functional as F

from .cmgan.discriminator import Discriminator
from .registry import ModelRegistry


@ModelRegistry.register
class GANWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.generator = ModelRegistry()[cfg.model.name, cfg.model]
        self.discriminator = Discriminator(16)

    def forward(self, x):
        return self.generate(x)

    def generate(self, x):
        comp = torch.complex(x[:, 0], x[:, 1])
        x = torch.stack([comp.abs(), comp.angle()], dim=1)
        return self.generator(x)

    def discriminate(self, x_clean, x_noisy, x_denoised):
        clean_mag = x_clean.norm(dim=1, keepdim=True)
        denoised_mag = x_denoised.norm(dim=1, keepdim=True)
        pred_1 = self.discriminator(clean_mag, clean_mag)
        pred_2 = self.discriminator(clean_mag, denoised_mag)
        return torch.concat((pred_1, pred_2))

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        try:
            return super().load_state_dict(state_dict, strict, assign)
        except KeyError:
            self.generator.load_state_dict(state_dict, strict, assign)
