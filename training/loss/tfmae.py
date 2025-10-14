import torch
import torch.nn as nn
import torch.nn.functional as F


class TFMAELoss(nn.Module):
    def forward(self, x_denoised, x_clean, x_noisy):
        x_clean = x_clean.to(x_denoised.device)
        x_noisy = x_noisy.to(x_denoised.device).squeeze(1)

        X_cl = torch.stft(x_clean, 
                          512, 
                          256, 
                          512, 
                          torch.hamming_window(512, device=x_clean.device), 
                          return_complex=True)
        X_de = torch.stft(x_denoised,
                          512, 
                          256, 
                          512, 
                          torch.hamming_window(512, device=x_clean.device), 
                          return_complex=True)

        loss = F.l1_loss(torch.view_as_real(X_de), 
                         torch.view_as_real(X_cl))

        return loss, {'loss': loss.clone().detach().item()}
