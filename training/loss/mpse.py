import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import util

logger = logging.getLogger(__name__)


class MPSELoss(nn.Module):
    def __init__(self, cfg, mode, transforms=None):
        super().__init__()
        if mode == 'disc':
            self.call_fn = self._disc_fn
        elif mode == 'gen':
            self.call_fn = self._gen_fn
        else:
            raise ValueError

        self.loss_weights = cfg.loss_weights
        self.fs = cfg.fs

        self.transforms = transforms

    def _gen_fn(self, x_denoised, x_denoised_audio, x_clean, x_clean_audio, y_pred):
        y_pred = y_pred[x_clean.shape[0]:].squeeze()
        labels = torch.ones_like(y_pred)

        metric_loss = F.mse_loss(y_pred, labels)

        mae = F.l1_loss(x_denoised_audio.squeeze(1), x_clean_audio.to(x_denoised_audio.device).squeeze(1))

        clean_comp = torch.complex(x_clean[:, 0], x_clean[:, 1]) 
        clean_mag = clean_comp.abs()
        est_comp = torch.complex(x_denoised[:, 0], x_denoised[:, 1])
        mag_est = est_comp.abs()
        mag_loss = F.mse_loss(mag_est, clean_mag)

        clean_phase = clean_comp.angle()
        est_phase = est_comp.angle()
        phase_loss = phase_losses(clean_phase, est_phase)
        consistency = self.consistency(x_denoised, x_denoised_audio)

        ri_loss = 2 * F.mse_loss(x_denoised, x_clean)

        total_loss = ri_loss * self.loss_weights.ri + \
                        mag_loss * self.loss_weights.mag + \
                        mae * self.loss_weights.time + \
                        metric_loss * self.loss_weights.metric + \
                        phase_loss * self.loss_weights.phase + \
                        consistency * self.loss_weights.consist

        loss_dict = {'gen/pesq_mse': metric_loss.item(),
                     'gen/pesq_pred': y_pred.mean().item(),
                     'gen/mag_loss': mag_loss.item(),
                     'gen/ri_loss': ri_loss.item(),
                     'gen/time_loss': mae.item(),
                     'gen/loss': total_loss.item()}

        return total_loss, loss_dict

    def _disc_fn(self, x_denoised,  x_denoised_audio, x_clean, x_clean_audio, y_pred):
        batch_size = x_clean_audio.shape[0]
        # Max PESQ for clean-clean
        y_pred_clean = y_pred[:batch_size]
        clean_loss = F.mse_loss(y_pred_clean, torch.ones_like(y_pred_clean))

        y_pred_denoised = y_pred[batch_size:]
        true_pesq = util.batched_pesq(x_clean_audio.squeeze(1), x_denoised_audio, self.fs)
        denoised_loss = F.mse_loss(y_pred_denoised.squeeze(), true_pesq.to(y_pred_denoised.device))

        loss_dict = {'disc/clean_pesq_mse': clean_loss.item(), 
                     'disc/denoised_pesq_mse': denoised_loss.item(),
                     'disc/denoised_pesq': true_pesq.mean().item()}

        return clean_loss + denoised_loss, loss_dict

    def consistency(self, x_est, x_est_audio):
        out = []
        for x in x_est_audio:
            out.append(self.transforms(x[None])[0])

        x_est_tf = torch.stack(out)
        loss = 2 * F.mse_loss(x_est_tf, x_est)
        if torch.isnan(loss):
            logger.warning('Consistency loss is NaN')
            return torch.zeros_like(loss)
        return loss
        
    def forward(self, x_denoised, x_denoised_audio, x_clean, x_clean_audio, x_noisy, y_pred):
        return self.call_fn(x_denoised, x_denoised_audio, x_clean, x_clean_audio, y_pred)


def phase_losses(phase_r, phase_g):

    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))
    iaf_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))

    return ip_loss + gd_loss + iaf_loss

def anti_wrapping_function(x):

    return torch.abs(x - torch.round(x / (2 * torch.pi)) * 2 * torch.pi)
