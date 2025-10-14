import logging
import os

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EmbTrainer:
    def __init__(self, cfg, model, dataset, chkpt_mngr, log_mngr, device):
        """Trainer Class

        Class for creating embeddings

        Args:
            cfg (OmegaConf): Hydra config
            model (nn.Module): Model
            dataset (Iterable): torch DataLoader object
            device (torch.device): CUDA/CPU
        """
        self.cfg = cfg
        self.model = model
        self.dataset = dataset
        self.device = device

        self.save_path = self._get_save_path()

    @torch.no_grad()
    def run(self):
        iterator = tqdm(self.dataset) 
        for batch, ((x_clean_raw, x_clean), (x_noisy_raw, x_noisy, recon)) in enumerate(iterator):
            x_clean, x_noisy = x_clean.to(self.device), x_noisy.to(self.device)
            self.model(x_clean, x_noisy)
                
        if hasattr(self.model, 'finalize'):
            self.model.finalize()

        if self.cfg.training.save_embs:
            self.model.save_embs(self.save_path) 

        if self.cfg.training.save:
            self.model.save(self.save_path) 


    def _get_save_path(self):
        base_path = self.cfg.training.save_path
        
        if hasattr(self.cfg.data, 'testset'):
            dataset = self.cfg.data.testset.name
        else:
            dataset = self.cfg.data.dataset.name
        
        if hasattr(self.cfg.model, 'foundation'):
            model = self.cfg.model.foundation.model
        else:
            model = self.cfg.model.name

        file_base = f'{model}_{dataset}' + '_{type}.th'

        return os.path.join(base_path, file_base)
