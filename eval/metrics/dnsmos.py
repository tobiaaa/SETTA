import torch
from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore

from ._base import Metric


class DNSMOS(Metric):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.dnsmos = DeepNoiseSuppressionMeanOpinionScore(self.fs, False, device=device, num_threads=1) 

    def update(self, x_clean, x_noisy, x_denoised, kwargs):
        dnsmos = self.dnsmos(x_denoised)
        self.append(dnsmos)

        return dnsmos

    def names(self):
        return ['P808_MOS', 'DNSMOS_SIG', 'DNSMOS_BAK', 'DNSMOS_OVL']
