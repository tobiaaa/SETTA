import logging
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
import transformers

logger = logging.getLogger(__name__)

# Prevent CuFFT OOM
torch.backends.cuda.cufft_plan_cache[0].max_size = 0

class LaDen(nn.Module):
    def __init__(self, cfg, model, recon):
        super().__init__()
        self.model = model
        self.foundation = _get_model(cfg.foundation)
        self.recon_fn = recon.reconstruct
        self.transform = recon

        self.source_dict = deepcopy(model.state_dict())

        self.grad_accum = cfg.batch_size
        self._i = 0
        self._back = 0
        self.loss_threshold = cfg.loss_threshold
        self.alpha = cfg.ema_alpha
        self.consistency_weight = cfg.consistency_weight
        self.temp = cfg.power_temp
        
        self.A = torch.load('checkpoints/WavLM_EARS_map.th').to(torch.float32)

        if hasattr(self.model, 'get_adapt_groups'):
            parameters = self.model.get_adapt_groups(cfg.weights)
            adapt_params = sum((param.numel() for param in self.model.get_adapt_groups(cfg.weights)))        
        elif hasattr(self.model, 'get_param_groups'):
            parameters = self.model.get_param_groups()
            adapt_params = sum((param.numel() for param in self.model.get_param_groups()))        
        else:
            parameters = self.model.parameters()
            adapt_params = sum((param.numel() for param in self.model.parameters()))        

        self.opt = torch.optim.AdamW(parameters, cfg.lr)

        tot_params = sum((param.numel() for param in self.model.parameters()))        
        logger.info(f'Adapting {adapt_params}/{tot_params} Parameters')

    @torch.enable_grad()
    def forward(self, x, x_raw, recon):
        
        # Prediction
        with torch.no_grad():
            x_denoised = self.model.evaluate(x)
            x_denoised_audio_ret = self.recon_fn(x_denoised, recon)

        x_raw = x_raw[:, 0, :64_000]
        x, recon = self.transform(x_raw)
        x = x[None]
        x_denoised = self.model(x)
        x_denoised_audio = self.recon_fn(x_denoised, recon)

        x_raw = x_raw.squeeze(1)
        diff = x_denoised_audio.shape[1] - x_raw.shape[1]
        assert diff >= 0 and diff < 400, diff
        if diff != 0:
            x_denoised_audio = x_denoised_audio[:, :-diff]
        
        consistency = self.consistency(x_denoised_audio, x_raw)

        x_emb = self.embed(x_denoised_audio)
        noisy_emb = self.embed(x_raw)

        y_emb_est = noisy_emb @ self.A.T 
        y_emb_est = y_emb_est / y_emb_est.norm(dim=-1, keepdim=True)

        sim = (x_emb * y_emb_est).sum(dim=-1)
        loss = 1.0 - sim
        
        if loss <= self.loss_threshold:
            loss = loss + self.consistency_weight * consistency
            loss.backward()
            self._back += 1

        # Gradient Accumulation
        if (self._i + 1) % self.grad_accum == 0:
            self.opt.step()
            self.opt.zero_grad()
            self.apply_ema()
        
        self._i += 1

        output_dict = {'Loss': loss.item(), 'Rate': self._back / self._i}

        return x_denoised_audio_ret, loss, output_dict

    def _envelope(self, x):
        x = x.squeeze()
        x = AF.highpass_biquad(x, 16_000, 50)

        h = torch.arange(x.shape[-1], device=x.device)[None]
        h = torch.arange(-x.shape[-1], x.shape[-1], device=x.device)[None]
        h = (1 - torch.cos(torch.pi * h)) / (torch.pi * h + 1e-8)
        x_hil = AF.fftconvolve(x, h, 'same')[...,:x.shape[-1]]
        
        return x_hil.abs()

    def consistency(self, x_denoised, x_noisy):

        x_noisy = self.spectral_sub(x_noisy)

        x_denoised_frames = x_denoised.unfold(1, 1024, 512)
        x_noisy_frames = x_noisy.unfold(1, 1024, 512)

        denoised_env = self._envelope(x_denoised_frames)
        noisy_env = self._envelope(x_noisy_frames)

        denoised_env = denoised_env - torch.mean(denoised_env, dim=-1, keepdim=True)
        noisy_env = noisy_env - torch.mean(noisy_env, dim=-1, keepdim=True)
        
        # Compute correlation
        correlation = torch.sum(noisy_env * denoised_env, dim=-1) / (
            torch.norm(noisy_env, dim=-1) * torch.norm(denoised_env, dim=-1) + 1e-8
        )
        loss_weight = self.power_weight(x_denoised_frames)

        consistency= 1.0 - (correlation * loss_weight.squeeze()).sum(dim=0).mean()

        return consistency

    @torch.no_grad()
    def spectral_sub(self, x):
        X = torch.stft(torch.complex(x, torch.zeros_like(x)), 
                       512, 
                       256,
                       512,
                       torch.hamming_window(512, device=x.device))
        
        X, phase = X.abs(), X.angle()


        power = (X ** 2).sum(1)
        _, idx = torch.topk(power, int(0.05 * power.shape[-1]), largest=False)

        noise_freq = X[...,idx.squeeze()].mean(-1, keepdim=True)
        X = X - noise_freq
        X = F.relu(X)

        X = torch.complex(X * torch.cos(phase), X * torch.sin(phase))

        x_recon = torch.istft(X,
                              512,
                              256,
                              512,
                              torch.hamming_window(512, device=X.device),
                              onesided=False)

        diff = x.shape[-1] - x_recon.shape[-1]

        x_recon = F.pad(x_recon, [0, diff])
        return x_recon


    @torch.no_grad()
    def power_weight(self, x):
        power = (x ** 2).sum(dim=-1, keepdim=True)
        
        weights = F.softmax(power / self.temp, 1)
        return weights[0]


    def embed(self, x):
        x = x.to(self.foundation.dtype).squeeze(1)
        output = self.foundation(x) 
        x_feat = output['extract_features'].mean(dim=1)
        x_feat = x_feat / x_feat.norm(dim=-1, keepdim=True)
        return x_feat


    def apply_ema(self):
        new_dict = self.model.state_dict()

        for key, val in new_dict.items():
            new_dict[key] = self.alpha * val + (1 - self.alpha) * self.source_dict[key]

        self.model.load_state_dict(new_dict)


def _get_model(name):
    if name == 'WavLM':
        model = transformers.WavLMModel.from_pretrained("microsoft/wavlm-large",
                                                        torch_dtype=torch.float32)
    elif name == 'Wav2Vec':
        model = transformers.Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", 
                                                           torch_dtype=torch.float32)
    else:
        raise NameError(f'Model "{name}" unknown')

    return model
