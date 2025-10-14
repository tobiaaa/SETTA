import logging

import torch

from eval.evaluator import (DAEvaluator, EmbeddingEvaluator, SEEvaluator, async_metric_manager)

from . import metrics

logger = logging.getLogger(__name__)

ASYNC_METRICS = {
    "PESQ",
    "STOI",
    "Composite",
    "SSNR",
    "SISDR",
    "WER",
    "EMBSIM",
    "LLR",
    "CD",
    "FWSEG",
    "SRMR",
}

def get_metrics(cfg, device):
    metric_list = []
    for name in cfg.metrics:
        metric_list.append(_get_metric(name, cfg, device))

    return metric_list


def _get_metric(name, cfg, device):
    if cfg.async_metrics:
        device = torch.device('cpu')
        if name not in ASYNC_METRICS:
            raise NotImplementedError(f'Metric {name} currently does not support async evaluation')
    
    if name is None:
        return None
    
    try:
        cls = getattr(metrics, name)
    except AttributeError:
        raise ModuleNotFoundError(f'Metric "{name}" not found. Choose from {[cls.__name__ for cls in metrics.__all__]}')

    return cls(cfg, device)


def get_evaluator(cfg, model, test_loader, adaptation, metrics, device, logger=None):
    if cfg.eval.name == 'SE':
        return SEEvaluator(cfg, model, test_loader, metrics, device)
    elif cfg.eval.name == 'Emb':
        return EmbeddingEvaluator(cfg, model, test_loader, metrics, device) 
    elif cfg.eval.name == 'DA':
        return DAEvaluator(cfg, model, test_loader, adaptation, metrics, device, logger) 
    else:
        raise ValueError(f'Evaluator "{cfg.eval.name}" unknown')
