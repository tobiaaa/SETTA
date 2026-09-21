import logging
import math

import torch
import torch.nn as nn

from .util import MMSEEstimator, SampleCache

logger = logging.getLogger(__name__)


class BLINC(nn.Module):
    def __init__(self, cfg, model, recon, device):
        super().__init__()
        self.model = model
        self.device = device

        if cfg.model in ['CMGAN', 'CMGANLite', 'MPSENet', 'CMGANTest', 'MPSENetTest']:
            self.get_norm_fn = _norm_comp
        elif cfg.model in ['AmplitudeMasking']:
            self.get_norm_fn = _norm_slice
        else:
            logger.warning('Unknown model, assuming norm via slice')
            self.get_norm_fn = _norm_slice

        self.power_compress = cfg.power_compress if cfg.power_compress is not None else 1.0

        self.recon_fn = recon.reconstruct
        self.transform = recon
        self.fs = cfg.fs

        self.per_exp = 2.0 / self.power_compress

        hop = next((t.hop for t in self.transform.transforms
                    if hasattr(t, 'hop')), 100)
        self.mmse = MMSEEstimator(cfg.blind, hop / self.fs)

        self.regress = cfg.regress

        self.max_length = cfg.max_length

        self.mode = cfg.mode
        if self.mode not in ('deploy', 'cache', 'cache_eval'):
            raise ValueError(f'Mode "{self.mode}" unknown')

        self._i = 0

        self.cache = SampleCache(cfg.cache_path, self.mode, type(model).__name__)

    @torch.no_grad()
    def forward(self, x, x_raw, recon):
        if self.mode == 'cache_eval':
            return self._forward_cache_eval(x_raw)

        per = self.get_norm_fn(x)[0].float() ** self.per_exp
        blind = self.mmse.submit(per.cpu().numpy())

        mask, ctx = self._extract(x)

        mask_flat = mask.flatten()
        rank = mask_flat.argsort()
        n = mask_flat.numel()
        q = torch.linspace(0, 1, n, device=self.device)

        snr_db, act_frac = blind.result()

        if self.mode == 'cache':
            self.cache.add(rank, mask.shape, ctx, recon,
                           {'snr_db': snr_db, 'act_frac': act_frac},
                           x_raw.shape[-1])
            pred_wav = self._reconstruct_mask(mask[None], ctx, recon)[0]
            return pred_wav[None], torch.tensor(0.0), {}

        a, s, d = self._regress_params(act_frac)
        mask = self._build_mask((a, s, d), q, rank, mask.shape)
        best_wav = self._reconstruct_mask(mask, ctx, recon)[0]

        best_wav = self._loudness_renorm(best_wav, x_raw, snr_db)

        metrics = {'a': a, 's': s, 'd': d,
                   'snr_db': snr_db, 'act_frac': act_frac}

        return best_wav[None], torch.tensor(0.0), metrics

    @torch.no_grad()
    def _extract(self, x):
        self.model.return_parts = True
        if self.max_length is None or x.shape[-1] <= self.max_length:
            mask, mag, phase, comp = self.model(x)
        else:
            mask, mag, phase, comp = self._extract_chunked(x)
        self.model.return_parts = False
        ctx = {'mag': mag[0],
               'cos': torch.cos(phase[0]), 'sin': torch.sin(phase[0]),
               'cr': comp[0, 0] if comp is not None else None,
               'ci': comp[0, 1] if comp is not None else None}
        return mask[0], ctx

    @torch.no_grad()
    def _extract_chunked(self, x):
        n_frames = x.shape[-1]
        size = self.max_length // 2
        x = nn.functional.pad(x, (0, -n_frames % size))
        parts = zip(*(self.model(w) for w in x.split(size, dim=-1)))
        return [None if p[0] is None else torch.cat(p, dim=-1)[..., :n_frames]
                for p in parts]

    def _line(self, coef, x):
        val = coef.slope * x + coef.icpt
        if coef.get('log', False):
            val = math.exp(val)
        return min(max(val, coef.min), coef.max)

    def _regress_params(self, act_frac):
        a = self._line(self.regress.a, act_frac)
        d = self._line(self.regress.d, act_frac)
        s = self._line(self.regress.s, act_frac)
        return a, s, d

    def _loudness_renorm(self, wav, x_raw, snr_db):
        snr_lin = 10 ** (snr_db / 10)
        power_in = torch.sum(x_raw ** 2)
        power_out = torch.sum(wav ** 2).clamp(min=1e-12)
        power_target = power_in * snr_lin / (1.0 + snr_lin)
        return wav * torch.sqrt(power_target / power_out)

    def _build_mask(self, params, q, rank, mask_shape):
        n = q.shape[0]
        mask = torch.empty(n, device=self.device)
        f = self._warp(params, q).clamp(0, 1)
        mask[rank] = f
        return mask.reshape(1, *mask_shape)

    def _warp(self, p, q):
        a, s, d = p
        return (1 - d) * torch.sigmoid(a * (q - s)) + d

    @torch.no_grad()
    def _reconstruct_mask(self, mask, ctx, recon):
        x_mag = ctx['mag'][None] * mask
        real = x_mag * ctx['cos']
        imag = x_mag * ctx['sin']
        if ctx['cr'] is not None:
            real = real + ctx['cr']
            imag = imag + ctx['ci']
        x_out = torch.stack([real, imag], dim=1)
        return self.recon_fn(x_out, recon)

    @torch.no_grad()
    def _forward_cache_eval(self, x_raw):
        rank, mask_shape, ctx, recon, feats = self.cache.next(x_raw.shape[-1],
                                                               self.device)
        q = torch.linspace(0, 1, rank.numel(), device=self.device)

        snr_db = feats.get('snr_db_scale', feats['snr_db'])

        a, s, d = self._regress_params(feats['act_frac'])
        mask = self._build_mask((a, s, d), q, rank, mask_shape)
        best_wav = self._reconstruct_mask(mask, ctx, recon)[0]
        best_wav = self._loudness_renorm(best_wav, x_raw, snr_db)

        metrics = {'a': a, 's': s, 'd': d,
                   'snr_db': snr_db, 'act_frac': feats['act_frac']}
        return best_wav[None], torch.tensor(0.0), metrics


def _norm_slice(x):
    return x[:, 0]


def _norm_comp(x):
    return x.norm(dim=1)
