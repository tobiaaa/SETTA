import logging
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

logger = logging.getLogger(__name__)


class RemixIT(nn.Module):
    def __init__(self, cfg, model, transforms):
        super().__init__()
        self.student = model
        self.transforms = transforms

        self.teacher_dict = model.state_dict()
        self.teacher = deepcopy(model)

        self.batch_size = cfg.batch_size
        self._i = 0
        self.teacher_update = cfg.teacher_update
        self.alpha = cfg.alpha
        
        if hasattr(self.student, 'get_adapt_groups'):
            parameters = self.student.get_adapt_groups(cfg.weights)
            adapt_params = sum((param.numel() for param in self.student.get_adapt_groups(cfg.weights)))        
        elif hasattr(self.student, 'get_param_groups'):
            parameters = self.student.get_param_groups()
            adapt_params = sum((param.numel() for param in self.student.get_param_groups(cfg.weights)))        
        else:
            parameters = self.student.parameters()
            adapt_params = sum((param.numel() for param in self.student.parameters()))        

        self.opt = torch.optim.AdamW(parameters, cfg.lr)
        tot_params = sum((param.numel() for param in self.student.parameters()))        
        logger.info(f'Adapting {adapt_params}/{tot_params} Parameters')
        self.clean_buffer = []
        self.noise_buffer = []
        self._last_loss = torch.zeros(())
        self.gen = torch.Generator().manual_seed(123)

    def forward(self, x, x_raw, recon):
        if hasattr(self.teacher, 'evaluate'):
            x_denoised = self.teacher.evaluate(x)
        else:
            x_denoised = self.teacher(x)
        x_denoised_audio = self.transforms.reconstruct(x_denoised, recon)
        
        diff = x_denoised_audio.shape[1] - x_raw.squeeze(1).shape[1]
        assert diff >= 0 and diff < 400, diff
        if diff != 0:
            x_denoised_audio = x_denoised_audio[:, :-diff]

        # HACK: Truncate for OOM
        self.clean_buffer.append(x_denoised_audio[:, :32_000])
        self.noise_buffer.append((x_raw.squeeze(1) - x_denoised_audio)[:, :32_000])

        if (self._i + 1) % self.batch_size == 0:
            self.step()
        if (self._i + 1) % self.teacher_update == 0:
            self.apply_ema()

        self._i += 1
            
        return x_denoised_audio, self._last_loss

    @torch.enable_grad()
    def step(self):
        x_clean = self.make_batch(self.clean_buffer)
        x_noise = self.make_batch(self.noise_buffer)

        # Shuffle noise
        x_noise = x_noise[torch.randperm(x_noise.shape[0], generator=self.gen)]
        x_noisy = x_clean + x_noise

        x_noisy_tr = []
        recon_buff = []
        for x in x_noisy:
            x_tr, recon = self.transforms(x[None])
            x_noisy_tr.append(x_tr)
            recon_buff.append(recon)
        x_noisy_tr = torch.stack(x_noisy_tr)

        # Adapt
        x_denoised = self.student(x_noisy_tr)
        recon = torch.utils.data.default_collate(recon_buff)
        x_denoised_audio = self.transforms.reconstruct(x_denoised, recon)

        diff = x_denoised_audio.shape[1] - x_clean.squeeze(1).shape[1]
        assert diff >= 0 and diff < 400, diff
        if diff != 0:
            x_denoised_audio = x_denoised_audio[:, :-diff]
        
        self.opt.zero_grad()
        loss = F.mse_loss(x_denoised_audio, x_clean)
        loss.backward()
        self.opt.step()

        self._last_loss = loss.detach()

        self.clean_buffer = []
        self.noise_buffer = []


    def make_batch(self, buffer):
        assert len(buffer) <= self.batch_size
        lengths = [x.shape[-1] for x in buffer]
        min_length = min(lengths)
        def crop(x):
            if x.shape[-1] > min_length:
                return x[..., :min_length]
            else:
                return x
        buffer = [crop(x) for x in buffer]
        batch = torch.concat(buffer)
        return batch


    def apply_ema(self):
        new_dict = self.student.state_dict()

        for key, val in new_dict.items():
            new_dict[key] = self.alpha * val + (1 - self.alpha) * self.teacher_dict[key]

        self.teacher.load_state_dict(new_dict)
