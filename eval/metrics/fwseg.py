"""
Adapted from REVERB challenge MATLAB Code
"""
from math import ceil, log2
import logging

import torch
import torchaudio.functional as AF
import torch.nn.functional as F

from ._base import Metric

logger = logging.getLogger(__file__)


class FWSEG(Metric):
    """
    Frequency weighted SNRseg
    """
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.order = 23

        self.win_len = int(0.025 * self.fs)
        self.shift = int(0.01 * self.fs)
        self.num_crit = 25
        self.window = torch.hamming_window(self.win_len, device=self.device)
        self.n_fft = 2 ** ceil(log2(self.win_len))

        self.mel_mat = AF.melscale_fbanks(self.n_fft // 2 + 1,
                                          0.0,
                                          self.fs / 2,
                                          self.order,
                                          self.fs)
        self.mel_mat = self.mel_mat.to(self.device)

    @torch.no_grad()
    def update(self, x_clean, x_noisy, x_denoised, kwargs):
        x_clean = x_clean.squeeze(dim=1).to(self.device)
        x_denoised = x_denoised.squeeze(dim=1).to(self.device)

        update_vals = []
        # Iterate over batch
        for x_cl, x_de in zip(x_clean, x_denoised):
            fwseg = self._fwseg(x_cl, x_de)
            fwseg = fwseg.mean()

            self.append(fwseg)
            update_vals.append(fwseg)

        return torch.tensor(update_vals)

    def _fwseg(self, x_clean, x_denoised):

        x_clean = x_clean / x_clean.norm(dim=-1, keepdim=True)
        x_denoised = x_denoised / x_denoised.norm(dim=-1, keepdim=True)

        X_cl = torch.stft(F.pad(x_clean, (0, 112)),
                          self.n_fft,
                          self.shift,
                          self.win_len,
                          self.window,
                          center=False,
                          return_complex=True).abs()

        X_de = torch.stft(F.pad(x_denoised, (0, 112)),
                          self.n_fft,
                          self.shift,
                          self.win_len,
                          self.window,
                          center=False,
                          return_complex=True).abs()

        X_cl = X_cl.T @ self.mel_mat
        X_de = X_de.T @ self.mel_mat

        W = X_cl ** 0.2
        E = X_de - X_cl

        ds = 10 * (W * (X_cl ** 2 / (E ** 2 + 1e-8)).log10()).sum(1, keepdim=True) / W.sum(1, keepdim=True) 
        ds = ds.clamp(min=-10, max=35)

        return ds

