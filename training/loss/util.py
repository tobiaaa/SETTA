import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
from joblib import Parallel, delayed
from pesq import PesqError, pesq

from training.loss.energy import EnergyConservingLoss
from training.loss.multi_stft import MultiSTFTLoss
from training.loss.pesq import PESQLoss
from training.loss.si_sdr import SISDRLoss
from training.loss.ssra import SSRALoss
from training.loss.tfmae import TFMAELoss

from .metric_gan import MetricGANLoss
from .mpse import MPSELoss
from .segan import SEGANLoss

warnings.simplefilter('ignore', category=FutureWarning)

def get_loss(name, cfg, device=None, **kwargs):
    if name == 'MSE':
        return LossWraper(nn.MSELoss())
    elif name == 'MAE':
        return LossWraper(nn.L1Loss())
    elif name == 'BCE':
        return LossWraper(nn.BCEWithLogitsLoss())
    elif name == 'GAN_disc':
        return SEGANLoss(cfg, mode='disc')
    elif name == 'GAN_gen':
        return SEGANLoss(cfg, mode='gen')
    elif name == 'metricGAN_disc':
        return MetricGANLoss(cfg, mode='disc')
    elif name == 'metricGAN_gen':
        return MetricGANLoss(cfg, mode='gen')
    elif name == 'MPSE_disc':
        return MPSELoss(cfg, mode='disc')
    elif name == 'MPSE_gen':
        return MPSELoss(cfg, mode='gen', **kwargs)
    elif name == 'multiSTFT':
        return MultiSTFTLoss(cfg)
    elif name == 'Energy':
        return EnergyConservingLoss()
    elif name == 'SISDR':
        return SISDRLoss()
    elif name == 'PESQ':
        return PESQLoss(cfg)
    elif name == 'SSRA':
        return SSRALoss(cfg, device)
    elif name == 'TFMAE':
        return TFMAELoss()
    elif name.startswith('Compound:'):
        return CompoundLoss(name, cfg, device)
    else:
        raise ValueError(f'Loss function "{name}" unknown')


class LossWraper(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, x_denoised, x_clean, x_noisy):
        loss = self.loss_fn(x_denoised, x_clean.to(x_denoised.device))

        return loss, {'loss': loss.item()}


class CompoundLoss(nn.Module):
    def __init__(self, name, cfg, device):
        super().__init__()
        name = ':'.join(name.split(':')[1:])
        name = name.replace(' ', '')
        components = name.split('+')
        self.factors = []
        self.loss_fn = []
        self.names = []
        for comp in components:
            if '*' in comp:
                factor, loss = comp.split('*')
                self.factors.append(float(factor))
                self.loss_fn.append(get_loss(loss, cfg, device))
                self.names.append(loss)
            else:
                self.factors.append(1.0)
                self.loss_fn.append(get_loss(comp, cfg, device))
                self.names.append(comp)

    def forward(self, x_denoised, x_clean, x_noisy):
        loss = 0
        loss_dict = {}
        for factor, loss_fn, name in zip(self.factors, self.loss_fn, self.names):
            new_loss, new_dict = loss_fn(x_denoised, x_clean, x_noisy)
            loss_dict |= {f'{name}/{key}': val for key, val in new_dict.items()}
            loss = loss + factor * new_loss         
        loss_dict['loss'] = loss.clone().detach().item()
        return loss, loss_dict

def istft(x_re, x_img, n_fft=400, hop=100):
    x_comp = torch.complex(x_re, x_img)
    x_time = torch.istft(x_comp,
                         n_fft,
                         hop,
                         window=torch.hamming_window(n_fft, device=x_re.device),
                         onesided=True)
    return x_time


def batched_pesq(x_clean, x_denoised, fs):
    score = Parallel(-1)(delayed(pesq_fn)(clean, denoised, fs)
                         for clean, denoised in zip(x_clean,
                                                    x_denoised))
    return torch.tensor(score) / 5


def pesq_fn(x_clean, x_denoised, fs):
    if fs != 16_000:
        x_clean = AF.resample(x_clean, fs, 16_000)
        x_denoised = AF.resample(x_denoised, fs, 16_000)

    return pesq(16_000,
                x_clean.clone().detach().cpu().numpy(),
                x_denoised.clone().detach().cpu().numpy(),
                'wb',
                on_error=PesqError.RETURN_VALUES)
