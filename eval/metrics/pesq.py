import pesq
import torch
import torchaudio.functional as AF

from ._base import Metric


class PESQ(Metric):
    def update(self, x_clean, x_noisy, x_denoised, kwargs):
        update_vals = []
        # Iterate over batch
        for x_cl, x_deno in zip(x_clean, x_denoised):
            if self.fs != 16_000:
                x_cl = AF.resample(x_cl, self.fs, 16_000)
                x_deno = AF.resample(x_deno, self.fs, 16_000)
                
            score = pesq.pesq(16_000,
                              x_cl.squeeze().cpu().numpy(),
                              x_deno.squeeze().cpu().numpy())
            self.append(torch.tensor(score))
            update_vals.append(score)
        return torch.tensor(update_vals)
