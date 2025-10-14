import logging
import os

import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from . import dns, ears, ears_demand, transforms, voicebank, voicebank_wham

logger = logging.getLogger(__name__)

def get_dataloader(cfg):
    """Get Dataset + Dataloader from config"""
    transforms = get_transforms(cfg)
    if 'trainset' in cfg:
        train_cfg = cfg.trainset
        test_cfg = cfg.testset
    else:
        train_cfg = cfg.dataset
        test_cfg = cfg.dataset

    # Train
    _, dataloader_train = _get_dataloader(train_cfg, cfg, 'train', transforms)

    # Test
    _, dataloader_test = _get_dataloader(test_cfg, cfg, 'test', transforms)

    return dataloader_train, dataloader_test


def _get_dataloader(data_cfg, cfg, split, transforms):
    """Get Dataset + Dataloader from config"""
    if data_cfg.name == 'VoiceBank':
        # Train
        dataset = voicebank.VoiceBankDemand(cfg,
                                            data_cfg,
                                            split=split,
                                            transforms=transforms)
        dataloader = _dataloader(cfg,
                                 dataset,
                                 split=split)

    elif data_cfg.name == 'EARS':

        dataset = ears.EARSWHAM(cfg,
                                data_cfg,
                                split=split,
                                transforms=transforms)

        dataloader = _dataloader(cfg,
                                 dataset,
                                 split=split)

    elif data_cfg.name == 'EARS_DEMAND':
        dataset = ears_demand.EARS_DEMAND(cfg,
                                          data_cfg,
                                          split=split,
                                          transforms=transforms)

        dataloader = _dataloader(cfg,
                                 dataset,
                                 split=split)

    elif data_cfg.name == 'DNS':
        dataset = dns.DNS(cfg,
                          data_cfg,
                          split=split,
                          transforms=transforms)

        dataloader = _dataloader(cfg,
                                 dataset,
                                 split=split)

    elif data_cfg.name == 'VoiceBankWHAM':
        dataset = voicebank_wham.VoiceBankWHAM(cfg,
                                               data_cfg,
                                               split=split,
                                               transforms=transforms)

        dataloader = _dataloader(cfg,
                                 dataset,
                                 split=split)

    else:
        raise ValueError(f'Dataset {data_cfg.name} unknown')

    return dataset, dataloader


def _dataloader(cfg, dataset, split='train'):
    """Function to get a DataLoader instance from config
    Args:
        cfg (OmegaConf): Training namespace of hydra config
        dataset (torch Dataset): Dataset instance
        split (str): train/test
    """

    n_workers = cfg.n_workers
    if n_workers is None:
        n_workers = len(os.sched_getaffinity(0))

    logger.info(f'Constructing Data Loader with {n_workers} workers')

    batch_size = cfg.batch_size if split == 'train' else cfg.test_batch_size
    shuffle = cfg.shuffle and split == 'train'
    if os.environ.get('FORCE_SHUFFLE', 'False') == 'True':
        logger.info('Forcing data shuffle')
        shuffle = True
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=n_workers,
                            prefetch_factor=cfg.n_prefetch,
                            shuffle=shuffle)
    return dataloader


def get_transforms(cfg):
    """Get Transforms from config"""
    if cfg.transforms is None or len(cfg.transforms) == 0:
        return nn.Identity()

    transforms = []
    reconstructions = []
    for name in cfg.transforms:
        out = get_transform(name, cfg)
        if isinstance(out, (list, tuple)):
            transform, recon = out
        else:
            transform = out
            recon = lambda x: x

        transforms.append(transform)
        reconstructions.append(recon)

    return TransformChain(transforms)


def get_transform(name, cfg):
    
    # If name starts with "*", do not reconstruct
    reconstruct = True
    if name.startswith('*'):
        reconstruct = False
        name = name.lstrip('*')

    if name == 'norm':
        return transforms.Norm(reconstruct)
    elif name == 'power_norm':
        return transforms.PowerNorm(reconstruct)
    elif name == 'loud_norm':
        return transforms.LoudNorm(reconstruct, cfg.fs)
    elif name == 'stft':
        return TransformWrapper(torchaudio.transforms.Spectrogram(), reconstruct)
    elif name == 'mel_stft':
        return TransformWrapper(torchaudio.transforms.MelSpectrogram(cfg.fs), reconstruct)
    elif name == 'log':
        return TransformWrapper(transforms.Log(), reconstruct)
    elif name == 'identity':
        return TransformWrapper(nn.Identity(), reconstruct)
    elif name.startswith('TF'):
        return transforms.Complex(name, cfg.stft, power_compress=cfg.power_compress, reconstruct=reconstruct)
    else:
        raise NameError(f'Transform "{name}" unknown')


class TransformWrapper(nn.Module):
    def __init__(self, module, reconstruct):
        super().__init__()
        self.module = module
        self.reconstruct_en = reconstruct

    def forward(self, x):
        output = self.module(x)
        if isinstance(output, (list, tuple)):
            return output
        else:
            return output, {}

    def reconstruct(self, x, **kwargs):
        if not self.reconstruct_en:
            return x

        if hasattr(self.module, 'reconstruct'):
            return self.module.reconstruct(x, **kwargs)
        else:
            return x


class TransformChain(nn.Module):
    def __init__(self, transforms):
        super().__init__()

        self.transforms = transforms

    def reconstruct(self, x, kwargs_list):
        assert len(self.transforms) == len(kwargs_list)
        for transform, kwargs in zip(reversed(self.transforms), reversed(kwargs_list)):
            x = transform.reconstruct(x, **kwargs)
    
        return x

    def forward(self, x):
        recon_kwargs = []
        for transform in self.transforms:
            x, recon_kwarg = transform(x)
            recon_kwargs.append(recon_kwarg)

        return x, recon_kwargs


def _identity(*args, **kwargs):
    return args, {}
