import logging
import os

import hydra
import torch
import torchaudio
from tqdm import tqdm

from .async_metric_manager import AsyncMetricManager
from .metric_manager import MetricManager

logger = logging.getLogger(__name__)


class DAEvaluator:
    def __init__(self, cfg, model, dataset, adaptation, metrics, device, logger):
        """Evaluator class to run evaluation routine for domain adaptation

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
        self.adaptation = adaptation

        self.base_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.denoised_path = os.path.join(self.base_path, 'denoised')

        if cfg.async_metrics:
            self.metric_manager = AsyncMetricManager(self.metrics)
        else:
            self.metric_manager = MetricManager(self.metrics)

        if hasattr(self.dataset.dataset, 'meta_names'):
            self.metric_manager.columns += self.dataset.dataset.meta_names

        self.logger = logger

        self.metrics_df = None

    @torch.no_grad()
    def run(self, save=False):
        if save:
            os.mkdir(self.denoised_path)
        
        self.dataset.dataset.return_file = True
        self.dataset.dataset.return_text = True

        iterator = tqdm(self.dataset, desc='Evaluation')
        for i, (clean, noisy, meta) in enumerate(iterator):
            x_clean_raw, x_clean = clean
            x_noisy_raw, x_noisy, recon = noisy
            filename = meta['file']

            x_clean, x_noisy = x_clean.to(self.device), x_noisy.to(self.device)
            x_clean_raw, x_noisy_raw = x_clean_raw.to(self.device), x_noisy_raw.to(self.device)
            x_denoised, loss, *metric_dict = self.adaptation(x_noisy,
                                                             x_noisy_raw,
                                                             recon)

            x_denoised = x_denoised.squeeze(1)

            # Match lengths - can slightly change in istft(stft(x))
            x_clean_raw = x_clean_raw.squeeze(1)
            diff = x_denoised.shape[1] - x_clean_raw.shape[1]
            assert diff >= 0 and diff < 400, diff
            if diff != 0:
                x_denoised = x_denoised[:, :-diff]

            self.metric_manager.update(x_clean_raw,
                                       x_noisy_raw, 
                                       x_denoised,
                                       filename[0],
                                       meta)

            if not metric_dict:
                self.log({'Loss': loss.item()}, i)
                iterator.set_postfix({'Loss': loss.item()})
            else:
                self.log(metric_dict[0], i)
                iterator.set_postfix(metric_dict[0])

            if save:
                file_name_cat = filename[0].replace('/', '#')
                torchaudio.save(os.path.join(self.denoised_path, file_name_cat),
                                x_denoised.cpu().squeeze(1), self.cfg.data.fs)
            
        self.metrics_df = self.metric_manager.get_df()
        if save:
            self.metrics_df.to_csv(os.path.join(self.denoised_path, 'metrics.csv'), index=False)
            file_name = upload_eval_data(self.base_path)
            log_stats(self.metrics_df.mean(numeric_only=True), file_name, self.cfg)
        
        self.logger.log_results(self.metrics_df.mean(numeric_only=True).to_dict())

    def log(self, metrics, i):
        if self.logger is None:
            return
        if i % self.logger.freq:
            return

        self.logger.log_metrics(metrics, i)


    def update_metrics(self, x_clean, x_noisy, x_denoised):
        results_dict = {}
        for metric in self.metrics:
            result = metric(x_clean, x_noisy, x_denoised).squeeze()
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
