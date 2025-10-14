import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

SPLIT_DICT = {
    'train': 'training',
    'test': 'test'
}

LANGUAGE_DICT = {
    'english': 'read_speech',
    'german': 'german_speech',
    'french': 'french_speech',
    'russian': 'russian_speech',
    'spanish': 'spanish_speech',
    'italian': 'italian_speech',
}


class DNS(Dataset):
    def __init__(self, cfg, data_cfg=None, split='train', transforms=None):
        """Deep Noise Suppression Challenge 2022 dataset

        Dataset class to load segments from the DNS 2022 pre-mixed dataset 

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
        assert split in ('train', 'test')
        self.top_dir = os.path.join(self.top_dir, SPLIT_DICT[split])
        self.split = split

        self.files = []
        for language in data_cfg.languages:
            self.files += self.get_lang_files(language)

        if data_cfg.shuff_lang:
            np.random.default_rng(123).shuffle(self.files)

        self.fs = cfg.fs

        self.sample_len = int(self.fs * cfg.sample_length)
        self.max_len = data_cfg.max_len * self.fs

        self.eval_mode = self.split == 'test'
        self.return_file = False
        self.transforms = transforms

    def get_lang_files(self, language):
        lang_dir = os.path.join(self.top_dir, LANGUAGE_DICT[language])
        clean_dir = os.path.join(lang_dir, 'clean')
        noisy_dir = os.path.join(lang_dir, 'noisy')
        
        files = []
        for noisy_file in os.listdir(noisy_dir):
            id = noisy_file.rstrip('.wav').split('_')[-1]
            clean_file = f'clean_fileid_{id}.wav'
            if not os.path.exists(os.path.join(clean_dir, clean_file)):
                print('Clean file does not exist')
                continue
            files.append((os.path.join(clean_dir, clean_file),
                          os.path.join(noisy_dir, noisy_file)))
        
        if self.split == 'test':
            files = files[:500]
        return files

    def __getitem__(self, index):
        
        clean_file, noisy_file = self.files[index]

        x_clean, Fs_clean = torchaudio.load(clean_file)
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

        if x_clean.shape[1] > self.max_len:
            x_clean = x_clean[:, :self.max_len]
            x_noisy = x_noisy[:, :self.max_len]
        
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

    def __len__(self):
        return len(self.files)
