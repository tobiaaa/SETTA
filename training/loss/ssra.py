import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


class SSRALoss(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.emb_model = transformers.WavLMModel.from_pretrained("microsoft/wavlm-large",
                                                                 torch_dtype=torch.float32)

        self.emb_model = self.emb_model.to(device)

    def forward(self, x_denoised, x_clean, x_noisy):
        x_clean = x_clean.to(x_denoised.device)
        x_noisy = x_noisy.to(x_denoised.device)
        
        emb_clean = self._get_emb(x_clean)
        emb_denoised = self._get_emb(x_denoised)

        sim = (emb_clean * emb_denoised).sum(dim=1).mean()
        loss = 1.0 - sim
        
        return loss, {'loss': loss.clone().detach().item()}

    def _get_emb(self, x):
        output = self.emb_model(x) 
        x_feat = output['extract_features'].mean(dim=1)
        x_feat = x_feat / x_feat.norm(dim=-1, keepdim=True)
        return x_feat

