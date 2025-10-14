import logging
import os

import hydra
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SEEvaluator:
    def __init__(self, cfg, model, dataset, metrics, device):
        """Evaluator class to run evaluation routine for speech enhancement

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

        self.base_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.denoised_path = os.path.join(self.base_path, 'denoised')

        self.metrics_df = None

    @torch.no_grad()
    def run(self, save=False):
        os.mkdir(self.denoised_path)
        
        metric_results = []
        columns = ['Filename'] + sum([metric.names() for metric in self.metrics], [])
        if hasattr(self.dataset.dataset, 'meta_names'):
            columns += self.dataset.dataset.meta_names

        self.dataset.dataset.return_file = True

        iterator = tqdm(self.dataset, desc='Evaluation')
        self.model.eval()
        for i, (clean, noisy, meta) in enumerate(iterator):
            x_clean_raw, x_clean = clean
            x_noisy_raw, x_noisy, recon = noisy
            filename = meta['file']
            x_clean, x_noisy = x_clean.to(self.device), x_noisy.to(self.device)
            x_clean_raw, x_noisy_raw = x_clean_raw.to(self.device), x_noisy_raw.to(self.device)
            if hasattr(self.model, 'evaluate'):
                x_denoised = self.model.evaluate(x_noisy)
            else:
                x_denoised = self.model(x_noisy)

            x_denoised = self.dataset.dataset.transforms.reconstruct(x_denoised, recon)
            x_denoised = x_denoised.squeeze(1)

            # Match lengths - can slightly change in istft(stft(x))
            x_clean_raw = x_clean_raw.squeeze(1)
            diff = x_denoised.shape[1] - x_clean_raw.shape[1]
            assert diff >= 0 and diff < 400, diff
            if diff != 0:
                x_denoised = x_denoised[:, :-diff]

            metric_dict = self.update_metrics(x_clean_raw, x_noisy_raw, x_denoised, meta)
            metric_dict['Filename'] = filename[0]
            metric_dict |= self.extract_meta(meta)
              
            metric_results.append(metric_dict)
            if save:
                torchaudio.save(os.path.join(self.denoised_path, filename[0]),
                                x_denoised.cpu().squeeze(1), self.cfg.data.fs)

        self.metrics_df = pd.DataFrame(metric_results, columns=columns)
        if save:
            self.metrics_df.to_csv(os.path.join(self.denoised_path, 'metrics.csv'), index=False)

    def update_metrics(self, x_clean, x_noisy, x_denoised, kwargs):
        results_dict = {}
        for metric in self.metrics:
            result = metric(x_clean, x_noisy, x_denoised, kwargs).squeeze()
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

    def extract_meta(self, meta):
        meta_dict = {}
        for key, val in meta.items():
            if key in ['file']:
                continue
            if isinstance(val, torch.Tensor):
                meta_dict[key] = val.item()
            elif isinstance(val, list):
                meta_dict[key] = val[0]
            else:
                meta_dict[key] = val

        return meta_dict

    def __repr__(self) -> str:

        output = [metric.__repr__() for metric in self.metrics]

        return "\n".join(output)
