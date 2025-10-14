import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import ModelRegistry


@ModelRegistry.register
class EmbeddingApproximator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        registry = ModelRegistry()
        self.foundation = registry[cfg.foundation.name, cfg.foundation]
        self.approximator = registry[cfg.approximator.name, cfg.approximator]

    def forward(self, x_clean, x_noisy):
        clean_emb = self.foundation(x_clean)
        noisy_emb = self.foundation(x_noisy)
        
        self.approximator(clean_emb, noisy_emb)

    def evaluate(self, x_clean, x_noisy):
        x_clean_emb = self.foundation(x_clean)
        x_noisy_emb = self.foundation(x_noisy)

        return x_clean_emb, x_noisy_emb, self.approximator.evaluate(x_clean_emb, x_noisy_emb)

    def finalize(self):
        self.approximator.finalize()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self.approximator.load_state_dict(state_dict, strict, assign)

    def save_embs(self, path):
        self.approximator.save_embs(path)

    def save(self, path):
        self.approximator.save(path)
