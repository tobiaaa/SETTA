import datetime
import inspect
import logging
import os

import wandb
from omegaconf import OmegaConf

from .log_helper import LogHelper, require_enabled

logger = logging.getLogger(__name__)


class WandB(LogHelper):
    def __init__(self, cfg, cfg_tot):
        super().__init__(cfg, cfg_tot)
        self.enabled = os.environ.get('WANDB_ENABLED', default='False') == 'True'

        if not self.enabled:
            logger.warning('Not logging to weights&biases')
            return

        os.environ['WANDB_SILENT'] =  'True'

        name = datetime.datetime.now().__format__('%y/%m/%d %H-%M-%S')

        wandb.login(key=os.environ['WANDB_API'])
        self.run = wandb.init(entity=os.environ['WANDB_USER'],
                              project=os.environ['WANDB_PROJECT'],
                              tags=self.tags,
                              name=name,
                              job_type=os.environ['MODE'],
                              config=OmegaConf.to_container(cfg_tot, resolve=True),
                              dir=os.path.abspath('.wandb/'))


    @require_enabled
    def log_config(self, cfg):
        return

    @require_enabled
    def log_metrics(self, metrics_dict, step=None):
        self.run.log(metrics_dict, step)

    @require_enabled
    def log_results(self, metrics_dict):
        self.log_metrics(metrics_dict)
    
    @require_enabled
    def log_class_code(self, method):
        self.run.save(inspect.getfile(method.__class__))
