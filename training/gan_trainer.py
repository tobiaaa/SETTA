import torch
import torch.nn as nn
from tqdm import tqdm

from .loss import get_loss
from .util import get_lr_sched


class GanTrainer:
    def __init__(self, cfg, model, dataset, chkpt_mngr, log_mngr, device):
        """Trainer Class for GAN training

        Supports separate generator and discriminator losses and optimizers

        Args:
            cfg (OmegaConf): Hydra config
            model (nn.Module): Must implement `generate()` and `discriminate()`
            dataset (Iterable): Training dataloader
            chkpt_mngr (CheckpointManager): CheckpointManager object
            log_mngr: Logging manager (e.g. Neptune wrapper)
            device (torch.device): CUDA/CPU
        """
        self.cfg = cfg
        self.model = model
        self.dataset = dataset
        self.device = device
        self.chkpt_mngr = chkpt_mngr
        self.log_mngr = log_mngr
        self.grad_norm = cfg.training.grad_clip_norm

        self.gen_crit = get_loss(cfg.training.loss + '_gen', cfg.training, device, transforms=self.dataset.dataset.transforms)
        self.disc_crit = get_loss(cfg.training.loss + '_disc', cfg.training, device)

        self.gen_opt = torch.optim.AdamW(self.model.generator.parameters(),
                                         cfg.training.gen_lr,
                                         betas=(cfg.training.adam.beta_1,
                                                cfg.training.adam.beta_2))
        self.gen_sched = get_lr_sched(cfg.training.gen_lr_sched, self.gen_opt)

        self.disc_opt = torch.optim.AdamW(self.model.discriminator.parameters(),
                                          cfg.training.disc_lr,
                                          betas=(cfg.training.adam.beta_1,
                                                 cfg.training.adam.beta_2))
        self.disc_sched = get_lr_sched(cfg.training.disc_lr_sched, self.disc_opt)

    def run(self):
        self.model.train()
        loss_dict = {}
        log_dict = {}
        step = 0
        for epoch in range(self.cfg.training.epochs):
            iterator = tqdm(self.dataset,
                            desc=f'Epoch: {epoch+1}/{self.cfg.training.epochs}')

            for batch, ((x_clean_raw, x_clean), (x_noisy_raw, x_noisy, recon)) in enumerate(iterator):
                x_clean, x_noisy = x_clean.to(self.device), x_noisy.to(self.device)
                x_denoised = self.model(x_noisy)
                x_denoised_audio = self.dataset.dataset.transforms.reconstruct(x_denoised, recon)
                y_pred = self.model.discriminate(x_clean, x_noisy, x_denoised)

                # Generation step
                gen_loss, gen_dict = self.gen_crit(x_denoised,
                                                   x_denoised_audio,
                                                   x_clean,
                                                   x_clean_raw,
                                                   x_noisy, 
                                                   y_pred)
                self.gen_opt.zero_grad()
                gen_loss.backward()
                if self.grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.generator.parameters(), self.grad_norm)
                self.gen_opt.step()
                loss_dict['Gen Loss'] = gen_loss.item()
                log_dict = log_dict | loss_dict | gen_dict
                iterator.set_postfix(loss_dict)

                # Discrimination step
                y_pred = self.model.discriminate(
                    x_clean, x_noisy, x_denoised.detach())
                disc_loss, disc_dict = self.disc_crit(x_denoised,
                                                      x_denoised_audio,
                                                      x_clean,
                                                      x_clean_raw,
                                                      x_noisy, 
                                                      y_pred)
                self.disc_opt.zero_grad()
                disc_loss.backward()
                if self.grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), self.grad_norm)
                self.disc_opt.step()

                loss_dict['Disc Loss'] = disc_loss.item()
                log_dict = log_dict | loss_dict | disc_dict
                iterator.set_postfix(loss_dict)
                if step % self.log_mngr.freq == 0 and step != 0:
                    self.log_mngr.log_metrics(log_dict, step=step)
                    self.log_mngr.log_metrics({'training/gen_lr': self.gen_sched.get_last_lr()[0],
                                               'training/disc_lr': self.disc_sched.get_last_lr()[0]},
                                              step=step)

                step += 1

            self.gen_sched.step()
            self.disc_sched.step()

            if epoch % self.chkpt_mngr.freq == 0 and epoch != 0:
                self.chkpt_mngr.save_model(self.model.state_dict(), epoch=epoch)
        self.chkpt_mngr.save_model(self.model.state_dict())
