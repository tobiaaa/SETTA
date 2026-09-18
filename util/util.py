import json
import logging
import os
import warnings
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from .log_helper import LogHelper
from .wandb_helper import WandB

logger = logging.getLogger(__file__)


def get_logger(cfg, cfg_tot):
    if cfg.name == 'WandB':
        return WandB(cfg, cfg_tot)
    else:
        raise NameError(f'Experiment logger "{cfg.name}" unknown')


def format_results(df, metrics):
    metric_names = sum([metric.names() for metric in metrics], [])

    meta_columns = df.columns.to_list()

    meta_columns = list(set(meta_columns) - set(metric_names) - set(['Filename']))
    if not meta_columns:
        logger.warning('No meta columns. Skipping detailed results')
        return

    df_org = df.groupby(meta_columns).mean(numeric_only=True)

    return df_org


class DummyLogger:
    def __init__(self):
        self.freq = 1000
        return

    def log_config(self, cfg):
        return

    def log_metrics(self, metrics_dict, step=None):
        return

    def log_results(self, metrics_dict):
        return

    def log_class_code(self, method):
        return


class TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


logging.TqdmLoggingHandler = TqdmLoggingHandler
logging.captureWarnings(True)
if os.environ.get('SE_DEBUG', 'False') != 'True':
    warnings.simplefilter('once')


def if_resolver(condition, true_val, false_val):
    return true_val if condition.lower() == 'true' else false_val


OmegaConf.register_new_resolver("if", if_resolver)

if os.environ.get('SE_DEBUG', 'False') != 'True':
    torch._logging.set_logs(graph_breaks=False,
                            dynamo=logging.ERROR,
                            autograd=logging.ERROR)

    _ignore_warnings_path = Path(__file__).parent / 'ignore_warnings.json'
    with open(_ignore_warnings_path) as _f:
        for _pattern in json.load(_f):
            warnings.filterwarnings('ignore', message=_pattern)
else:
    torch.autograd.set_detect_anomaly(True)


_KNOWN_SE_VARS = {
    'SE_DEBUG',
    'SE_TRAINING_RUN',
    'SE_FORCE_SHUFFLE',
    'SE_FIX_SHUFFLE',
    'SE_FIXED_START',
    'SE_EMB_DIR',
    'SE_FORCE_TQDM',
    'SE_DISABLE_TQDM',
    'SE_PROG_INFO_F',
    'SE_PROG_TIME_F',
    'SE_PROG_TIME_ALPHA',
}


def check_env_vars():
    for key in os.environ:
        if key.startswith('SE_') and key not in _KNOWN_SE_VARS:
            logger.warning(f'Unknown SE_ environment variable "{key}" — possible typo?')
