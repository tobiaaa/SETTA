import torch
import torch.nn.functional as F
import torchaudio.functional as AF

# HACK: Gammatone is not installed properly
# Clone from https://github.com/detly/gammatone and change path
import sys
from pathlib import Path
sys.path.append(str(Path.home() / 'Developer' / 'Gammatone'))
try:
    import gammatone
except ImportError:
    raise ImportError('Gammatone needs to be installed manually')


from torchmetrics.audio import srmr

from ._base import Metric

C = 10 * (2 ** 0.5) / torch.log(torch.tensor(10))


class SRMR(Metric):
    """
    Speech to Reverberatin Modulation Energy Ratio
    """
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.srmr = srmr.SpeechReverberationModulationEnergyRatio(self.fs)

    def update(self, x_clean, x_noisy, x_denoised, kwargs):
        x_clean = x_clean.squeeze(dim=1)
        x_denoised = x_denoised.squeeze(dim=1)

        update_vals = []
        # Iterate over batch
        for x_cl, x_de in zip(x_clean, x_denoised):
            srmr_out = self.srmr(x_de)

            self.append(srmr_out)
            update_vals.append(srmr_out)

        return torch.tensor(update_vals)

