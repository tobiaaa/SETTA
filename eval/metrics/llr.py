"""
Adapted from 'Speech Enhancement: Theory and Practice', MATLAB Code

Not identical to REVERB challenge
"""
import logging

import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from ._base import Metric

logger = logging.getLogger(__file__)


class LLR(Metric):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.objective = 'down'

        if self.fs < 10_000:
            self.order = 10
        else:
            self.order = 16

        self.win_len = round(30 * self.fs / 1000)
        self.skip_rate = self.win_len // 4
        window = torch.arange(1, self.win_len + 1) / (self.win_len + 1)
        window = 0.5 * (1 - torch.cos(2 * torch.pi * window))
        self.window = window.to(self.device)

    def update(self, x_clean, x_noisy, x_denoised, kwargs):
        x_clean = x_clean.squeeze(dim=1).to(self.device)
        x_denoised = x_denoised.squeeze(dim=1).to(self.device)

        update_vals = []
        # Iterate over batch
        for x_cl, x_de in zip(x_clean, x_denoised):
            llr = self._llr(x_cl, x_de)            
            llr, _ = torch.topk(llr, int(0.95 * llr.shape[0]), dim=0, largest=False)
            llr = llr.mean()

            self.append(llr)
            update_vals.append(llr)

        return torch.tensor(update_vals)

    def _llr(self, x_clean, x_denoised):

        x_clean_fr = x_clean.unfold(-1, self.win_len, self.skip_rate) 
        x_denoised_fr = x_denoised.unfold(-1, self.win_len, self.skip_rate) 

        x_clean_fr = x_clean_fr * self.window[None]
        x_denoised_fr = x_denoised_fr * self.window[None]

        R_cl, Ref_cl, A_cl = self._lpcoeff(x_clean_fr)
        R_de, Ref_de, A_de = self._lpcoeff(x_denoised_fr)

        numerator = A_de[:, None, :] @ self._toeplitz(R_cl) @ A_de[:, :, None]
        denominator = A_cl[:, None, :] @ self._toeplitz(R_cl) @ A_cl[:, :, None]

        distortion = torch.minimum(torch.ones_like(numerator) * 2, torch.log(numerator / denominator))

        return distortion.squeeze((1, 2))


    def _lpcoeff(self, x):
        x_pad = F.pad(x.flip(-1), (0, self.order))
        corr = AF.convolve(x, x_pad, 'valid')

        E = torch.zeros_like(corr)
        E[:, 0] = corr[:, 0]
        a = torch.zeros_like(corr)
        r_coeff = torch.zeros_like(corr)

        for i in range(self.order):
            a_past = a.clone()
            sum_term = torch.sum(a_past[:, :i] * corr[:, 1:i+1].flip(-1), dim=-1)
            r_coeff[:, i] = (corr[:, i+1] - sum_term) / E[:, i]
            a[:, i] = r_coeff[:, i].clone() 
            a[:, :i] = a_past[:, :i] - r_coeff[:,i][:, None] * a_past[:, :i].flip(-1)
            E[:, i+1] = (1 - r_coeff[:, i] ** 2) * E[:, i]

        return corr, r_coeff, F.pad(-a[:, :-1], (1, 0), value=1.0)


    def _toeplitz(self, x):
        d = x.shape[-1]
        x_pad = F.pad(x, (d-1, 0), 'reflect')
        toep = x_pad.unfold(-1, d, 1).flip(-1)

        return toep
