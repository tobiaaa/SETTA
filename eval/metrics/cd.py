"""
Adapted from REVERB challenge MATLAB Code
"""
from math import ceil, log, log2

import torch

from ._base import Metric


class CD(Metric):
    """
    Cepstrum Distance Metric
    """
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.objective = 'down'

        self.order = 24
        self.win_size = int(0.025 * self.fs)
        self.shift_size = int(0.01 * self.fs)

        self.window = torch.hann_window(self.win_size, device=self.device)

    def update(self, x_clean, x_noisy, x_denoised, kwargs):
        x_clean = x_clean.squeeze(dim=1).to(self.device)
        x_denoised = x_denoised.squeeze(dim=1).to(self.device)

        update_vals = []
        # Iterate over batch
        for x_cl, x_de in zip(x_clean, x_denoised):
            cep = self._unsync_cepstrum(x_cl, x_de)            

            self.append(cep)
            update_vals.append(cep)

        return torch.tensor(update_vals)

    def _unsync_cepstrum(self, x_clean, x_denoised):
        x_cl_fr = x_clean.unfold(-1, self.win_size, self.shift_size)
        x_de_fr = x_denoised.unfold(-1, self.win_size, self.shift_size) 

        x_cl_fr = x_cl_fr * self.window[None]
        x_de_fr = x_de_fr * self.window[None]

        cep_cl = self._cepstrum(x_cl_fr)[:, :self.order + 1]
        cep_de = self._cepstrum(x_de_fr)[:, :self.order + 1]

        cep_cl = cep_cl - cep_cl.mean(dim=0, keepdim=True)
        cep_de = cep_de - cep_de.mean(dim=0, keepdim=True)

        err = (cep_cl - cep_de) ** 2
        ds = 10 / log(10.0) * (2 * err[:, 1:].sum(-1) + err[:, 0]).sqrt()
        ds = ds.clamp(min=0.0, max=10.0)

        return ds.mean()


    def _cepstrum(self, x):
        Px = torch.fft.fft(x, 2 ** ceil(log2(x.shape[-1]))).abs()
        Px = Px.clamp(min=1e-5 * Px.max())

        c = torch.fft.ifft(Px.log()).real
        return c




