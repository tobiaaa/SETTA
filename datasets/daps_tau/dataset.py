import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset

from .create import get_index

logger = logging.getLogger(__name__)


class DAPS_TAU(Dataset):
    def __init__(self, cfg, data_cfg=None, split='train', transforms=None):
        """DAPS + TAU Dataset

        Dataset class to load segments from the DAPS + TAU dataset

        Args:
            cfg (OmegaConf): Complete config (# TODO: Change to only use dataset namespace)
            split (str): train/test
            transforms (Callable): Callable to transform unbatched segments
        """
        super().__init__()
        if data_cfg is None:
            self.top_dir = cfg.dataset.path
        else:
            self.top_dir = data_cfg.path
        assert split in ('train', 'val', 'test')
        self.clean_dir = Path(self.top_dir) / 'clean'
        self.noisy_dir = Path(self.top_dir) / 'noisy'

        self.split = split
        self.fs = cfg.fs
        self.sample_len = int(self.fs * cfg.sample_length)

        self.index = get_index(cfg, data_cfg)

        self.eval_mode = self.split == 'test'
        self.return_file = False
        self.transforms = transforms

        self.random_start = os.environ.get('SE_FIXED_START', 'False') == 'False'

        self.generator = torch.Generator().manual_seed(123)

        if os.environ.get('SE_FIX_SHUFFLE', 'False') == 'True':
            gen = torch.Generator().manual_seed(123)
            self.index_map = torch.randperm(len(self), generator=gen)
            logger.info('Using fixed shuffle')
        else:
            self.index_map = torch.arange(0, len(self))

    def __getitem__(self, index):
        index = self.index_map[index]
        row = self.index.iloc[index.item()]

        x_clean, Fs_clean = torchaudio.load(self.clean_dir / row['mix_name'])
        x_noisy, Fs_noisy = torchaudio.load(self.noisy_dir / row['mix_name'])

        if Fs_clean != self.fs:
            x_clean = AF.resample(x_clean, Fs_clean, self.fs)
        if Fs_noisy != self.fs:
            x_noisy = AF.resample(x_noisy, Fs_noisy, self.fs)

        pad_val = 0
        if not self.eval_mode:
            # Pad if necessary
            if x_clean.shape[1] <= self.sample_len:
                pad_val = self.sample_len - x_clean.shape[1] + 2
                x_clean = self._pad_end(x_clean, pad_val)
                x_noisy = self._pad_end(x_noisy, pad_val)
            # Select random segment
            if self.random_start:
                start = torch.randint(0, x_clean.shape[1] -
                                      self.sample_len, (), generator=self.generator)
            else:
                start = 0
            x_clean = x_clean[:, start:start + self.sample_len]
            x_noisy = x_noisy[:, start:start + self.sample_len]

        x_clean_tr, _ = self.transforms(x_clean)
        x_noisy_tr, noisy_recon = self.transforms(x_noisy)

        meta = {}

        if self.return_file:
            meta['file'] = row['mix_name']

        if meta:
            meta['snr'] = row['snr']
            meta['idx'] = index
            meta['pad'] = pad_val
            return (x_clean, x_clean_tr), (x_noisy, x_noisy_tr, noisy_recon), meta

        if self.return_file:
            return (x_clean, x_clean_tr), (x_noisy, x_noisy_tr, noisy_recon), row['mix_name']

        return (x_clean, x_clean_tr), (x_noisy, x_noisy_tr, noisy_recon)

    def __len__(self):
        return len(self.index)

    def _pad_end(self, x, pad_val):
        x = F.pad(x, (0, pad_val))
        return x
