import os
import logging
import torch
import torch.nn as nn
from .base import ApproxBase
from tqdm import tqdm

logger = logging.getLogger(__name__)

class NNMap(ApproxBase):
    def __init__(self):
        super().__init__()
        self.clean_embs = []
        self.noisy_embs = []
        self.model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        ) 

        self.model.to(torch.device('cuda'))
        self.opt = torch.optim.Adam(self.model.parameters(), 1e-3)

    def evaluate(self, x_clean, x_noisy):
        dtype = x_noisy.dtype
        x_clean_est = self.model(x_noisy.to(torch.float32))
        x_clean_est = x_clean_est / x_clean_est.norm(dim=-1, keepdim=True)
        x_clean_est = x_clean_est.to(dtype)
        return x_clean_est

    @torch.enable_grad()
    def finalize(self):
        clean_embeddings = torch.concat(self.clean_embs).to(torch.float32)
        noisy_embeddings = torch.concat(self.noisy_embs).to(torch.float32)
        clean_embeddings = clean_embeddings / clean_embeddings.norm(dim=-1, keepdim=True)
        noisy_embeddings = noisy_embeddings / noisy_embeddings.norm(dim=-1, keepdim=True)

        BATCH_SIZE = 512

        # Fit MLP
        iterator = tqdm(range(1_000_000), desc='MLP Fitting')
        for i in iterator:
            batch_idx = torch.randint(max(clean_embeddings.shape[0] - BATCH_SIZE, 1), ())
            x_clean = clean_embeddings[batch_idx:batch_idx+BATCH_SIZE]
            x_noisy = noisy_embeddings[batch_idx:batch_idx+BATCH_SIZE]
            x_pred = self.model(x_noisy)
            x_pred = x_pred / x_pred.norm(dim=-1, keepdim=True)
            
            loss = 1 - (x_pred * x_clean).sum(dim=-1).mean()
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            iterator.set_postfix({'Loss': loss.item()})
        
    def save(self, path):
        path = path.format(type='nn')
        if os.path.exists(path):
            logger.warning('Overwritting NN')
        
        torch.save(self.model.state_dict(), path)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.model.load_state_dict(state_dict)
