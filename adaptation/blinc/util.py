import atexit
import logging
import math
import os
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch

logger = logging.getLogger(__name__)
try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False
    logger.warning('numba unavailable')

    def njit(*args, **kwargs):
        return lambda fn: fn


CACHE_PATH = os.path.join(os.environ.get('SE_EMB_DIR', ''), 'BLINC_cache.pt')


class SampleCache:
    def __init__(self, path, mode, model_name):
        self.path = path or CACHE_PATH
        self.model_name = model_name
        self.entries = []
        self._i = 0

        if mode == 'cache':
            atexit.register(self.flush)
        elif mode == 'cache_eval':
            self.entries = torch.load(self.path, map_location='cpu')['entries']
            logger.info(f'Loaded {len(self.entries)} cached samples from {self.path}')

    def add(self, rank, mask_shape, ctx, recon, feats, n_raw):
        self.entries.append({
            'rank': rank.detach().cpu(),
            'mask_shape': tuple(int(s) for s in mask_shape),
            'mag': _nested_cpu(ctx['mag']), 'cos': _nested_cpu(ctx['cos']),
            'sin': _nested_cpu(ctx['sin']),
            'cr': _nested_cpu(ctx['cr']), 'ci': _nested_cpu(ctx['ci']),
            'recon': _nested_cpu(recon),
            'feats': {k: _nested_cpu(v) for k, v in feats.items()},
            'n_raw': int(n_raw),
        })

    def next(self, n_raw, device):
        entry = self.entries[self._i]
        if entry.get('n_raw', n_raw) != n_raw:
            raise RuntimeError(f'Cache misaligned at sample {self._i}: '
                               f'{entry["n_raw"]} vs {n_raw} samples, '
                               f'check SE_FIX_SHUFFLE and testset')
        self._i += 1

        rank = entry['rank'].to(device)
        ctx = {k: None if entry[k] is None else entry[k].to(device)
               for k in ('mag', 'cos', 'sin', 'cr', 'ci')}
        return rank, tuple(entry['mask_shape']), ctx, entry['recon'], entry['feats']

    def flush(self):
        if not self.entries:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save({'entries': self.entries,
                    'model': self.model_name,
                    'n': len(self.entries)}, self.path)
        logger.info(f'Cached {len(self.entries)} samples to {self.path}')


class MMSEEstimator:
    def __init__(self, cfg, shift):
        self.alpha_psd = math.exp(-shift / cfg.tau_psd)
        self.alpha_ph1 = math.exp(-shift / cfg.tau_ph1)
        self.xi_opt = 10 ** (cfg.xi_opt_db / 10)
        self.prior_ratio = cfg.prior_ratio

        # Init JIT
        self(np.ones((2, 8), dtype=np.float32))

        self._pool = (ThreadPoolExecutor(max_workers=1, thread_name_prefix='mmse')
                      if _HAVE_NUMBA else None)

    def __call__(self, periodogram):
        periodogram = np.ascontiguousarray(periodogram, dtype=np.float32)
        return _mmse_blind_feats(periodogram, self.alpha_psd, self.alpha_ph1,
                                 self.xi_opt, self.prior_ratio)

    def submit(self, periodogram):
        if self._pool is None:
            future = Future()
            future.set_result(self(periodogram))
            return future
        return self._pool.submit(self, periodogram)


def _nested_cpu(v):
    if torch.is_tensor(v):
        return v.detach().cpu()
    if isinstance(v, dict):
        return {k: _nested_cpu(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(_nested_cpu(x) for x in v)
    return v


@njit(cache=True, fastmath=True, nogil=True)
def _mmse_blind_feats(periodogram, alpha_psd, alpha_ph1, xi_opt,
                      prior_ratio=1.0, n_init=5):
    n_freq, n_frames = periodogram.shape
    n_lead = min(max(1, n_init), n_frames)

    glr_exp = xi_opt / (1.0 + xi_opt)
    log_lr_const = math.log(prior_ratio) - math.log1p(xi_opt)

    speech_sum = 0.0
    noise_sum = 0.0
    ph1_sum = 0.0

    for f in range(n_freq):
        # Initialise assuming the leading frames are noise-only.
        acc = 0.0
        for t in range(n_lead):
            acc += periodogram[f, t]
        noise_pow = max(acc / n_lead, 1e-12)
        ph1_mean = 0.5

        for t in range(n_frames):
            per = periodogram[f, t]
            snr_post = per / noise_pow
            ph1 = 1.0 / (1.0 + math.exp(-(log_lr_const + glr_exp * snr_post)))

            ph1_mean = alpha_ph1 * ph1_mean + (1.0 - alpha_ph1) * ph1
            if ph1_mean > 0.99 and ph1 > 0.99:
                ph1 = 0.99

            noise_est = ph1 * noise_pow + (1.0 - ph1) * per
            noise_pow = max(alpha_psd * noise_pow
                            + (1.0 - alpha_psd) * noise_est, 1e-12)

            if per > noise_pow:
                speech_sum += per - noise_pow
            noise_sum += noise_pow
            ph1_sum += ph1

    snr_lin = speech_sum / max(noise_sum, 1e-12)
    snr_db = 10.0 * math.log10(max(snr_lin, 1e-12))
    return snr_db, ph1_sum / (n_freq * n_frames)
