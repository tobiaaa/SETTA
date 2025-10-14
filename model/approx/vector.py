import os
import logging
import torch
import torch.nn as nn
from .base import ApproxBase

logger = logging.getLogger(__name__)

class VectorMap(ApproxBase):
    def __init__(self):
        super().__init__()
        self.clean_embs = []
        self.noisy_embs = []
        self.v = None

    def evaluate(self, x_clean, x_noisy):
        x_clean_est = x_noisy + self.v
        x_clean_est = x_clean_est / x_clean_est.norm(dim=-1, keepdim=True)
        return x_clean_est

    def finalize(self):
        clean_embeddings = torch.concat(self.clean_embs)
        noisy_embeddings = torch.concat(self.noisy_embs)
        
        self.v = (clean_embeddings - noisy_embeddings).mean(dim=0, keepdim=True)

    def save(self, path):
        if self.v is None:
            raise ValueError('Map not initialized, run `finalize()` first')
        path = path.format(type='vector')
        if os.path.exists(path):
            logger.warning('Overwritting map')
        
        torch.save(self.v, path)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.v = state_dict
