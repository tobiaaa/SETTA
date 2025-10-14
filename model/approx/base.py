from abc import ABC, abstractmethod
import os
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class ApproxBase(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        self.clean_embs = []
        self.noisy_embs = []

    def forward(self, x_clean, x_noisy):
        self.clean_embs.append(x_clean)
        self.noisy_embs.append(x_noisy)

    @abstractmethod
    def evaluate(self, x_clean, x_noisy):
        pass

    @abstractmethod
    def finalize(self):
        pass

    def save_embs(self, path):
        path = path.format(type='embeddings')

        clean_embeddings = torch.concat(self.clean_embs)
        noisy_embeddings = torch.concat(self.noisy_embs)
        
        torch.save({'clean': clean_embeddings,
                    'noisy': noisy_embeddings},
                   path)

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict, strict=True, assign=False):
        pass
