import torch

from ._base import Metric


class EmbSim(Metric):
    def update(self, x_clean, x_noisy, x_denoised, kwargs):

        x_clean = x_clean / x_clean.norm(dim=-1, keepdim=True)
        x_noisy = x_noisy / x_noisy.norm(dim=-1, keepdim=True)
        x_denoised = x_denoised / x_denoised.norm(dim=-1, keepdim=True)
        sim_base = (x_clean * x_noisy).sum()
        sim_est = (x_clean * x_denoised).sum()

        self.append(torch.tensor([sim_base, sim_est]))
        return torch.tensor([sim_base, sim_est])

    def names(self):
        return ['Baseline', 'Estimated']
