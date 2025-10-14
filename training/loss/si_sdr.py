import torch.nn as nn


class SISDRLoss(nn.Module):
    def forward(self, x_denoised, x_clean, x_noisy):
        x_clean = x_clean.to(x_denoised.device)
        x_noisy = x_noisy.to(x_denoised.device)
    
        scale = (x_clean * x_denoised).sum(1, keepdim=True) / ((x_clean ** 2).sum(1, keepdim=True) + 1e-8)

        sig = (scale * x_clean).norm(dim=1, keepdim=True)
        diff = (scale * x_clean - x_denoised).norm(dim=1, keepdim=True)

        loss = (sig / (diff + 1e-8)) ** 2
        loss = 10 * loss.clamp(min=1e-8).log10()
        loss = loss.mean()

        # Maximize SISDR -> Negative
        return -loss, {'loss': loss.clone().detach().item()}
