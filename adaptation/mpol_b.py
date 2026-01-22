import logging
from statistics import median

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mpol import MPol

logger = logging.getLogger(__name__)

class MPolBatched(MPol):

    @torch.enable_grad()
    def forward(self, x, x_raw, recon):
        
        # Prediction
        with torch.no_grad():
            self.model.return_mask = False
            x_denoised = self.model.evaluate(x)
            x_denoised_audio_ret = self.recon_fn(x_denoised, recon)

        # HACK: limit max length for OOM
        x_raw = x_raw[:, 0, :64_000]
        self.batch_buffer.append(x_raw)

        # Gradient Accumulation
        if (self._i + 1) % self.grad_accum == 0:
            self.step()
        
        self._i += 1

        output_dict = {'Loss': self.last_loss, 
                       'Sign': self.last_sign, 
                       'Wasser': self.last_wasser}

        return x_denoised_audio_ret, torch.tensor(self.last_loss), output_dict

    def step(self):
        batch = []
        targ_shape = int(median([x.shape[-1] for x in self.batch_buffer]))
        for x_raw in self.batch_buffer:
            # x, _ = self.transform(x_raw[..., :min_shape])
            if x_raw.shape[-1] > targ_shape:
                x_raw = x_raw[..., :targ_shape]
            else:
                diff = targ_shape - x_raw.shape[-1]
                if diff > x_raw.shape[-1]:
                    print("Skipping")
                    continue
                    
                x_raw = F.pad(x_raw, (0, targ_shape - x_raw.shape[-1]), 'circular')
            x, _ = self.transform(x_raw)
            batch.append(x)
        x = torch.stack(batch)
        self.model.return_mask = True
        x_est, mask = self.model(x)

        sign_loss = F.relu(-mask).mean(dim=(-1, -2)).mean()

        m_ss = torch.stack([self.spectral_sub(self.get_norm_fn(x)[i], x_est[i]) for i in range(x.shape[0])])
        wasser_loss = self.wasserstein(mask, m_ss)
        loss = self.sign_weight * sign_loss + self.wasser_weight * wasser_loss
        loss.backward()

        # Gradient Accumulation
        nn.utils.clip_grad_norm_(self.parameters, self.grad_norm)
        self.opt.step()
        self.opt.zero_grad()
        self.apply_ema()
        
        self.batch_buffer = []

        self.last_loss = loss.item()
        self.last_sign = sign_loss.item()
        self.last_wasser = wasser_loss.item()

        return 

