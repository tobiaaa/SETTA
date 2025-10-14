import os
import logging
import torch
import torch.nn as nn
from .base import ApproxBase

logger = logging.getLogger(__name__)

class AffineMap(ApproxBase):
    def __init__(self):
        super().__init__()
        self.clean_embs = []
        self.noisy_embs = []
        self.v = None
        self.map = None

    def evaluate(self, x_clean, x_noisy):
        x_clean_est = x_noisy @ self.map.T
        x_clean_est = x_clean_est / x_clean_est.norm(dim=-1, keepdim=True)

        x_clean_est = x_clean_est + self.v
        x_clean_est = x_clean_est / x_clean_est.norm(dim=-1, keepdim=True)

        return x_clean_est

    def finalize(self):
        clean_embeddings = torch.concat(self.clean_embs)
        noisy_embeddings = torch.concat(self.noisy_embs)
        
        noisy_inv = torch.linalg.pinv(noisy_embeddings.to(torch.float32).T)       
        self.map = clean_embeddings.T @ noisy_inv.to(clean_embeddings.dtype)

        clean_est = noisy_embeddings @ self.map.T
        clean_est = clean_est / clean_est.norm(dim=-1, keepdim=True)

        self.v = (clean_embeddings - clean_est).mean(dim=0, keepdim=True)
        

    def save(self, path):
        if self.v is None:
            raise ValueError('Map not initialized, run `finalize()` first')
        path = path.format(type='affine')
        if os.path.exists(path):
            logger.warning('Overwritting map')
        
        state_dict = {'vector': self.v, 'map': self.map}

        torch.save(state_dict, path)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.v = state_dict['vector']
        self.map = state_dict['map']
