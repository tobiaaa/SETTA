import logging
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MPol(nn.Module):
    def __init__(self, cfg, model, recon):
        super().__init__()
        self.model = model

        if cfg.model in ['CMGAN', 'CMGANLite', 'MPSENet', 'CMGANTest', 'MPSENetTest']:
            self.get_norm_fn = _norm_comp
        elif cfg.model in ['MiniMUCS']:
            self.get_norm_fn = _norm_slice
        else:
            logger.warning('Unknown model, assuming norm via slice')
            self.get_norm_fn = _norm_slice

        self.power_compress = cfg.power_compress if cfg.power_compress is not None else 1.0
        self.grad_norm = cfg.grad_norm

        self.recon_fn = recon.reconstruct
        self.transform = recon

        self.source_dict = deepcopy(model.state_dict())

        self.grad_accum = cfg.batch_size
        self.alpha = cfg.ema_alpha
        self.sign_weight = cfg.sign_weight
        self.wasser_weight = cfg.wasser_weight
        self._i = 0

        if hasattr(self.model, 'get_adapt_groups'):
            parameters = self.model.get_adapt_groups(cfg.weights)
            adapt_params = sum((param.numel()
                               for param in self.model.get_adapt_groups(cfg.weights)))
        elif hasattr(self.model, 'get_param_groups'):
            parameters = self.model.get_param_groups()
            adapt_params = sum((param.numel()
                               for param in self.model.get_param_groups(cfg.weights)))
        else:
            parameters = self.model.parameters()
            adapt_params = sum((param.numel() for param in self.model.parameters()))

        self.opt = torch.optim.AdamW(parameters, cfg.lr)

        tot_params = sum((param.numel() for param in self.model.parameters()))
        self.parameters = parameters
        logger.info(f'Adapting {adapt_params:,}/{tot_params:,} Parameters')

    @torch.enable_grad()
    def forward(self, x, x_raw, recon):

        # Prediction
        with torch.no_grad():
            self.model.return_mask = False
            x_denoised = self.model.evaluate(x)
            x_denoised_audio_ret = self.recon_fn(x_denoised, recon)

        # HACK: limit max length for OOM
        x_raw = x_raw[:, 0, :64_000]
        x, recon = self.transform(x_raw)
        x = x[None]
        self.model.return_mask = True
        x_est, mask = self.model(x)

        sign_loss = F.relu(-mask).mean()

        m_ss = self.spectral_sub(self.get_norm_fn(x)[0], x_est[0])
        wasser_loss = self.wasserstein(mask, m_ss)
        loss = self.sign_weight * sign_loss + self.wasser_weight * wasser_loss
        loss.backward()

        # Gradient Accumulation
        if (self._i + 1) % self.grad_accum == 0:
            nn.utils.clip_grad_norm_(self.parameters, self.grad_norm)
            self.opt.step()
            self.opt.zero_grad()
            self.apply_ema()

        self._i += 1

        output_dict = {'Loss': loss.item(),
                       'Sign': sign_loss.item(),
                       'Wasser': wasser_loss.item()}

        return x_denoised_audio_ret, loss, output_dict

    @torch.no_grad()
    def spectral_sub(self, x, x_est, n=32):
        x_est = x_est.norm(dim=0)
        _, min_idx = torch.topk(x_est.mean(0), n, dim=0, largest=False)
        noise_est = x[:, min_idx].mean(dim=1)

        x_est = x_est ** (1 / self.power_compress)
        noise_est = noise_est.clamp(min=1e-12) ** (1 / self.power_compress)

        gain = x_est / (x_est + noise_est[:, None]).clamp(min=1e-10)
        gain = gain.clamp(min=1e-10) ** self.power_compress

        return gain

    def wasserstein(self, mask, gain):
        mask = mask.reshape(-1)
        gain = gain.reshape(-1)

        loss = (mask.sort()[0] - gain.sort()[0]).abs().mean()
        return loss

    def apply_ema(self):
        new_dict = self.model.state_dict()

        for key, val in new_dict.items():
            new_dict[key] = self.alpha * val + (1 - self.alpha) * self.source_dict[key]

        self.model.load_state_dict(new_dict)


def _norm_slice(x):
    return x[:, 0]


def _norm_comp(x):
    return x.norm(dim=1)
