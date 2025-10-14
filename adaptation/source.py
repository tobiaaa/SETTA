import torch
import torch.nn as nn


class Source(nn.Module):
    def __init__(self, cfg, model, recon):
        super().__init__()
        self.model = model
        self.recon_fn = recon.reconstruct
        
    def forward(self, x, x_raw, recon):
        x_denoised = self.model.evaluate(x)
        x_denoised_audio = self.recon_fn(x_denoised, recon)
        return x_denoised_audio, torch.zeros(())
