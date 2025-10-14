import torch.nn as nn
import torch.nn.functional as F


class EnergyConservingLoss(nn.Module):
    def forward(self, x_denoised, x_clean, x_noisy):
        x_clean = x_clean.to(x_denoised.device)
        x_noisy = x_noisy.to(x_denoised.device)

        signal_loss = F.l1_loss(x_denoised, x_clean)

        # NOTE: Equivalent to signal_loss under additive noise 
        background_loss = F.l1_loss(x_noisy - x_denoised, x_noisy - x_clean)
        
        total_loss = signal_loss + background_loss

        loss_dict = {'loss': total_loss.item(),
                     'signal': signal_loss.item(),
                     'background': background_loss.item()}

        return total_loss, loss_dict
