import torch.nn as nn
from torch_pesq import PesqLoss


class PESQLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.loss_fn = PesqLoss(0.5, sample_rate=cfg.fs)

    def forward(self, x_denoised, x_clean, x_noisy):
        x_clean = x_clean.to(x_denoised.device)
        self.loss_fn = self.loss_fn.to(x_denoised.device)
    
        loss = self.loss_fn(x_clean, x_denoised).mean()

        # Maximize PESQ -> Negative
        return loss, {'loss': loss.item()}
