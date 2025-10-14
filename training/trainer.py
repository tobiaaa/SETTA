import torch
import torch.nn as nn
from tqdm import tqdm

from .loss import get_loss
from .util import get_lr_sched


class Trainer:
    def __init__(self, cfg, model, dataset, chkpt_mngr, log_mngr, device):
        """Trainer Class

        Class for running default end-to-end training

        Args:
            cfg (OmegaConf): Hydra config
            model (nn.Module): Model
            dataset (Iterable): torch DataLoader object
            chkpt_mngr (CheckpointManager): CheckpointManager object
            log_mngr: Logging manager (e.g. Neptune wrapper)
            device (torch.device): CUDA/CPU
        """
        self.cfg = cfg
        self.model = model
        self.dataset = dataset
        self.criterion = get_loss(cfg.training.loss, cfg.training, device)
        self.device = device
        self.chkpt_mngr = chkpt_mngr
        self.log_mngr = log_mngr

        if hasattr(self.model, 'get_param_groups'):
            parameters = self.model.get_param_groups()
        else:
            parameters = self.model.parameters()

        self.opt = torch.optim.AdamW(parameters,
                                     cfg.training.lr,
                                     betas=(cfg.training.adam.beta_1,
                                            cfg.training.adam.beta_2),
                                     weight_decay=cfg.training.l2_reg)

        self.sched = get_lr_sched(cfg.training.lr_sched, self.opt)
        self.grad_norm = cfg.training.grad_clip_norm

    def run(self):
        self.model.train()
        step = 0
        for epoch in range(self.cfg.training.epochs):
            iterator = tqdm(self.dataset,
                            desc=f'Epoch: {epoch+1}/{self.cfg.training.epochs}')
            for batch, ((x_clean_raw, x_clean), (x_noisy_raw, x_noisy, recon)) in enumerate(iterator):
                x_clean, x_noisy = x_clean.to(self.device), x_noisy.to(self.device)
                x_denoised = self.model(x_noisy)

                x_denoised = self.dataset.dataset.transforms.reconstruct(x_denoised, recon)

                loss, loss_dict = self.criterion(x_denoised.squeeze(1),
                                                 x_clean_raw.squeeze(1), 
                                                 x_noisy_raw)
                self.opt.zero_grad()
                loss.backward()
                if self.grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.opt.step()

                iterator.set_postfix(loss_dict)

                if step % self.log_mngr.freq == 0:
                    self.log_mngr.log_metrics({f'training/{key}': value for key, value in loss_dict.items()}, step=step)
                    self.log_mngr.log_metrics({'training/lr': self.sched.get_last_lr()[0]},
                                              step=step)

                step += 1

            self.sched.step()
            if epoch % self.chkpt_mngr.freq == 0 and epoch != 0:
                self.chkpt_mngr.save_model(self.model.state_dict(), epoch=epoch)

        self.chkpt_mngr.save_model(self.model.state_dict())
