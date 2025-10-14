import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiSTFTLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_fft = np.array((512, 1024, 2048))
        self.hops = np.array((50, 120, 240))
        self.windows = np.array((240, 600, 1200))

        self.n_losses = 3

    def forward(self, x_denoised, x_clean, x_noisy):
        
        x_clean = x_clean.to(x_denoised.device)
        
        mae = F.l1_loss(x_denoised, x_clean)
       
        spec_loss = torch.zeros_like(mae)
        mag_loss = torch.zeros_like(mae)
        
        for n_fft, hop, windowl in zip(self.n_fft, self.hops, self.windows):
            
            config = [n_fft,
                      hop,
                      windowl]
            
            x_den = torch.stft(x_denoised.squeeze(1), 
                               *config, 
                               window=torch.hamming_window(config[-1], device=x_denoised.device),
                               return_complex=True)
            x_den = torch.sqrt(torch.clamp(x_den.real**2+x_den.imag**2, min=1e-7))
            x_cl = torch.stft(x_clean.squeeze(1), 
                              *config, 
                              window=torch.hamming_window(config[-1], device=x_clean.device),
                              return_complex=True)
            x_cl = torch.sqrt(torch.clamp(x_cl.real**2+x_cl.imag**2, min=1e-7))

            spec_loss = spec_loss + self.spectral_convergence(x_den, x_cl)
            mag_loss = mag_loss + self.magnitude(x_den, x_cl)

        loss = spec_loss + mag_loss
        loss = loss / self.n_losses

        total_loss = mae + 0.1 * loss

        loss_dict = {'loss': total_loss.item(),
                     'spec': spec_loss.item(),
                     'mag': mag_loss.item(),
                     'mae': mae.item()}

        return total_loss, loss_dict

    def spectral_convergence(self, x_denoised, x_clean):
        numerator = torch.norm(x_denoised - x_clean, p='fro')
        denominator = torch.norm(x_clean, p='fro')
        loss = numerator / (denominator.clamp(min=1e-7))
        return loss

    def magnitude(self, x_denoised, x_clean):
        x_denoised = (x_denoised.clamp(min=1e-12)).log()
        x_clean = (x_clean.clamp(min=1e-7)).log()
        return F.l1_loss(x_denoised, x_clean)
