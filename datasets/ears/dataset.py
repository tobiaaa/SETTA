import glob
import logging
import os
import time

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class EARSWHAM(Dataset):
    def __init__(self, cfg, data_cfg=None, split='train', transforms=None):
        """EARS + WHAM Dataset

        Dataset class to load segments from the EARS + WHAM pre-mixed dataset

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
        self.clean_dir = os.path.join(self.top_dir, split, 'clean')
        self.noisy_dir = os.path.join(self.top_dir, split, 'noisy')

        self.fs = cfg.fs
        if cfg.fs != 48_000:
            logger.warning('Using EARS dataset at unusual sampling frequency')
            new_top = self.top_dir.replace('EARS', 'EARS_16')
            if cfg.fs == 16_000 and os.path.exists(new_top):
                logger.info('Switching to EARS_16')
                self.clean_dir = os.path.join(new_top, split, 'clean')
                self.noisy_dir = os.path.join(new_top, split, 'noisy')

        self.sample_len = int(self.fs * cfg.sample_length)

        self.files = sorted(glob.glob(os.path.join(self.clean_dir, '*', '*')))
        self.files = list(map(lambda x: os.path.join(*(x.split(os.sep)[-2:])), self.files))

        self.split = split
        self.eval_mode = self.split == 'test'
        self.return_file = False
        self.transforms = transforms

    def __getitem__(self, index):
        
        clean_file = self.files[index]
        noisy_file = self.get_noisy(clean_file)

        x_clean, Fs_clean = torchaudio.load(os.path.join(self.clean_dir, clean_file))
        x_noisy, Fs_noisy = torchaudio.load(noisy_file)

        if Fs_clean != self.fs:
            x_clean = AF.resample(x_clean, Fs_clean, self.fs)
        if Fs_noisy != self.fs:
            x_noisy = AF.resample(x_noisy, Fs_noisy, self.fs)

        if not self.eval_mode:
            # Pad if necessary
            if x_clean.shape[1] <= self.sample_len:
                pad_val = self.sample_len - x_clean.shape[1] + 2
                x_clean = F.pad(x_clean, (pad_val // 2, pad_val // 2, 0, 0))
                x_noisy = F.pad(x_noisy, (pad_val // 2, pad_val // 2, 0, 0))
            # Select random segment
            start = torch.randint(0, x_clean.shape[1] - self.sample_len, ())
            x_clean = x_clean[:, start:start+self.sample_len]
            x_noisy = x_noisy[:, start:start+self.sample_len]

        x_clean_tr, _ = self.transforms(x_clean)
        x_noisy_tr, noisy_recon = self.transforms(x_noisy)

        meta = {}

        if self.return_file:
            meta['file'] = clean_file.replace('/', '#')

        if meta:
            return (x_clean, x_clean_tr), (x_noisy, x_noisy_tr, noisy_recon), meta

        if self.return_file:
            return (x_clean, x_clean_tr), (x_noisy, x_noisy_tr, noisy_recon), clean_file

        return (x_clean, x_clean_tr), (x_noisy, x_noisy_tr, noisy_recon)

    def get_noisy(self, file):
        file, ext = os.path.splitext(file)
        
        noisy = glob.glob(os.path.join(self.noisy_dir, file + '_*' + ext))
        assert len(noisy) == 1
        return noisy[0]

    def __len__(self):
        return len(self.files)
