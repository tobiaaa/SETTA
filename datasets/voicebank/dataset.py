import os

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset

SUFFIX_DICT = {
    'train': ('clean_trainset_28spk_wav', 'noisy_trainset_28spk_wav', 'trainset_28spk_txt'),
    'test': ('clean_testset_wav', 'noisy_testset_wav', 'testset_txt')
}


class VoiceBankDemand(Dataset):
    def __init__(self, cfg, data_cfg=None, split='train', transforms=None):
        """VoiceBank + DEMAND Dataset

        Dataset class to load segments from the VoiceBank + DEMAND pre-mixed dataset

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
        clean_suff, noisy_suff, text_suff = SUFFIX_DICT[split]
        self.clean_dir = os.path.join(self.top_dir, clean_suff)
        self.noisy_dir = os.path.join(self.top_dir, noisy_suff)
        self.text_dir = os.path.join(self.top_dir, text_suff)

        self.fs = cfg.fs
        self.sample_len = int(self.fs * cfg.sample_length)

        self.files = sorted(os.listdir(self.clean_dir))
        self.split = split
        self.eval_mode = self.split == 'test'
        self.return_file = False
        self.return_text = False
        self._return_meta = False
        self.meta_data = None
        self.transforms = transforms

        self.gen = torch.Generator()

    def __getitem__(self, index):

        file = self.files[index]

        x_clean, Fs_clean = torchaudio.load(os.path.join(self.clean_dir, file))
        x_noisy, Fs_noisy = torchaudio.load(os.path.join(self.noisy_dir, file))

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
            start = torch.randint(0, x_clean.shape[1] - self.sample_len, (), generator=self.gen)
            x_clean = x_clean[:, start:start+self.sample_len]
            x_noisy = x_noisy[:, start:start+self.sample_len]

        x_clean_tr, _ = self.transforms(x_clean)
        x_noisy_tr, noisy_recon = self.transforms(x_noisy)

        meta = {}

        if self.return_file:
            meta['file'] = file

        if self.return_text:
            meta['text'] = self.load_text(file)

        if self.return_meta:
            meta |= self.get_meta(file) 

        if meta:
            return (x_clean, x_clean_tr), (x_noisy, x_noisy_tr, noisy_recon), meta

        return (x_clean, x_clean_tr), (x_noisy, x_noisy_tr, noisy_recon)

    def __len__(self):
        return len(self.files)

    def get_meta(self, file):
        assert self.meta_data is not None, 'Meta data not loaded'
        return self.meta_data[file.split('.')[0]]

    def load_meta_data(self):
        table_name = {'train': 'log_trainset_28k.txt', 'test': 'log_testset.txt'}
        self.meta_data = {}
        with open(os.path.join(self.top_dir, table_name[self.split]), 'r') as f:
            for line in f.readlines():
                file, noise, snr = line.split()
                self.meta_data[file] = {'noise': noise, 'snr': float(snr)}

    @property
    def return_meta(self):
        return self._return_meta

    @return_meta.setter
    def return_meta(self, value):
        self._return_meta = value
        if self.meta_data is None and value:
            self.load_meta_data()

    def load_text(self, file):
        file = file.split('.')[0] + '.txt'
        with open(os.path.join(self.text_dir, file), 'r') as f:
            text = f.read().strip()

        return text
