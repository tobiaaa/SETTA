import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF

from ..registry import ModelRegistry


@ModelRegistry.register
class Wiener(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_filter = cfg.n_filter
        self.n_est = cfg.n_est
        assert self.n_est >= self.n_filter

        self.pad_val = self.n_filter - 1

    @torch.no_grad
    def forward(self, x):
        
        # Estimate noise statistics
        x_noisy = x[..., :self.n_est].squeeze()
        r_nn = self.get_corr(x_noisy)

        R_yy = self.get_corr_mat(x.squeeze())
        
        r_dd = R_yy[0] - r_nn

        R_yy += 1e-4 * torch.eye(R_yy.shape[0], device=R_yy.device)
        w = torch.linalg.solve(R_yy, r_dd)

        x_in = F.pad(x, (self.pad_val, 0)).squeeze()
        x_denoised = AF.fftconvolve(x_in, w, mode='valid')

        return x_denoised[None, None]

    def get_corr(self, x):
        
        r_nn = x.unfold(-1, self.n_filter, self.n_filter // 2)
        r_nn = torch.cov(r_nn.T)

        r_nn = r_nn[0]

        return r_nn

    def get_corr_mat(self, x):
        x = x.unfold(-1, self.n_filter, self.n_filter)
        
        R = torch.cov(x.T)
        return R
