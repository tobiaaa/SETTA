import logging
from .neptune_helper import Neptune
from .wandb_helper import WandB
from .log_helper import LogHelper

logger = logging.getLogger(__file__)

def get_logger(cfg, cfg_tot):
    if cfg.name == 'WandB':
        return WandB(cfg, cfg_tot)
    elif cfg.name == 'Neptune':
        return Neptune(cfg, cfg_tot)
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
