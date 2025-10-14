import logging
import os

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset

from .index import get_index

logger = logging.getLogger(__name__)


class VoiceBankWHAM(Dataset):
    def __init__(self, cfg, data_cfg=None, split='train', transforms=None):
        """VoiceBank + WHAM Dataset

        Dataset class to load segments from the dataset

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

        self.fs = cfg.fs
        self.sample_len = int(self.fs * cfg.sample_length)

        self.split = split
        self.eval_mode = self.split == 'test'
        self.clean_dir = os.path.join(self.top_dir, 'clips')
        self.noise_dir = os.path.join(data_cfg.noise_path, 'high_res_wham', 'audio')

        self.index_df, self.noise_df = get_index(data_cfg, self.split, self.noise_dir)
        
        self.return_file = False
        self.transforms = transforms

        self.snr = data_cfg.snr


    def __getitem__(self, index):

        clean = self.index_df.iloc[index]
        noise = self.noise_df.iloc[clean['noise_idx']]

        clean_file = os.path.join(self.clean_dir, clean['filename'])
        noise_file = os.path.join(self.noise_dir, noise['file'])

        filename = '#'.join(clean['filename'].split('/')[-2:])

        Fs_clean = torchaudio.info(clean_file).sample_rate
        Fs_noise = torchaudio.info(noise_file).sample_rate

        x_clean, _ = torchaudio.load(clean_file, 
                                     num_frames=30 * Fs_clean)
        x_clean = x_clean.squeeze()

        x_noise, _ = torchaudio.load(noise_file, 
                                     num_frames=30 * Fs_noise,
                                     frame_offset=noise['start'] * (Fs_noise // 16_000))
        x_noise = x_noise[0]

        if Fs_clean != self.fs:
            x_clean = AF.resample(x_clean, Fs_clean, self.fs)
        if Fs_noise != self.fs:
            x_noise = AF.resample(x_noise, Fs_noise, self.fs)
        
        # Truncate
        x_noise = x_noise[:x_clean.shape[0]]
        x_clean = x_clean[:x_noise.shape[0]]

        # Combine noise and speech
        if self.snr is None:
            snr = clean['SNR']
        else:
            snr = self.snr
        target_loudness = clean['loudness'] - snr
        delta_loudness = target_loudness - noise['loudness']
        gain = 10.0 ** (delta_loudness / 20.0)
        noise_scaled = gain * x_noise
        x_noisy = x_clean + noise_scaled

        if not self.eval_mode:
            # Pad if necessary
            if x_clean.shape[0] <= self.sample_len:
                pad_val = self.sample_len - x_clean.shape[0] + 1
                x_clean = F.pad(x_clean, (pad_val // 2, pad_val // 2))
                x_noisy = F.pad(x_noisy, (pad_val // 2, pad_val // 2))
            # Select random segment
            start = torch.randint(0, x_clean.shape[0] - self.sample_len + 1, ())
            x_clean = x_clean[start:start+self.sample_len]
            x_noisy = x_noisy[start:start+self.sample_len]

        x_clean = x_clean[None]
        x_noisy = x_noisy[None]

        x_clean_tr, _ = self.transforms(x_clean)
        x_noisy_tr, noisy_recon = self.transforms(x_noisy)

        meta = {}

        if self.return_file:
            meta['file'] = filename

        if meta:
            meta['snr'] = clean['SNR']
            return (x_clean, x_clean_tr), (x_noisy, x_noisy_tr, noisy_recon), meta

        return (x_clean, x_clean_tr), (x_noisy, x_noisy_tr, noisy_recon)

    def __len__(self):
        return len(self.index_df)
