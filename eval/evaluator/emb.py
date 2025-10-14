import logging

import hydra
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

class EmbeddingEvaluator:
    def __init__(self, cfg, model, dataset, metrics, device):
        """Evaluator class to run evaluation routine for audio embeddings

        Args:
            cfg (OmegaConf): Complete config from hydra
            model (torch.nn.Module): Callable model to evaluate
            dataset (torch.utils.data.Dataset): Test dataset
            metrics (list[Metric]): List of callable metric classes
            device (torch.device): torch device
        """
        self.cfg = cfg
        self.model = model
        self.dataset = dataset
        self.metrics = metrics
        self.device = device

        if hasattr(self.cfg.data, 'trainset'):
            self.dataset_name = self.cfg.data.trainset.name
        else:
            self.dataset_name = self.cfg.data.dataset.name

        self.base_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        
    @torch.no_grad()
    def run(self, save=False):

        iterator = tqdm(self.dataset, desc='Evaluation')
        for i, (clean, noisy, *meta) in enumerate(iterator):
            x_clean_raw, x_clean = clean
            x_noisy_raw, x_noisy, recon = noisy
            x_clean, x_noisy = x_clean.to(self.device), x_noisy.to(self.device)

            x_clean_emb, x_noisy_emb, x_clean_est = self.model.evaluate(x_clean, x_noisy)

            self.update_metrics(x_clean_emb, x_noisy_emb, x_clean_est)


    def update_metrics(self, x_clean, x_noisy, x_denoised):
        results_dict = {}
        for metric in self.metrics:
            result = metric(x_clean, x_noisy, x_denoised, {}).squeeze()
            names = metric.names()
            if len(names) == 1:
                results_dict[names[0]] = result.item()
            else:
                for name, result in zip(names, result):
                    results_dict[name] = result.item()

        return results_dict

    def results(self):
        results = {metric.__class__.__name__: metric.result()
                   for metric in self.metrics}

        return results

    def result_dict(self):
        results_dict = {}
        for metric in self.metrics:
            names = metric.names()
            result = metric.result().squeeze()
            if len(names) == 1:
                results_dict[names[0]] = result.item()
            else:
                for name, result in zip(names, result):
                    results_dict[name] = result.item()

        return results_dict


    def __repr__(self) -> str:

        output = [metric.__repr__() for metric in self.metrics]

        return "\n".join(output)
