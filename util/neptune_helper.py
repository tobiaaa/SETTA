import logging
import os

import neptune
from neptune.integrations.python_logger import NeptuneHandler
from .log_helper import LogHelper, require_enabled

logger = logging.getLogger(__name__)


class Neptune(LogHelper):
    def __init__(self, cfg, cfg_tot):
        super().__init__(cfg, cfg_tot)
        self.freq = cfg.freq
        self.enabled = os.environ.get('NEPTUNE_ENABLED', default='False') == 'True'

        root_logger = logging.getLogger()

        if not self.enabled:
            logger.warning('Not logging to neptune')
            return

        if cfg.description is not None:
            description = cfg.description
        else:
            description = ""

        self.run = neptune.init_run(project=os.environ['NEPTUNE_PROJECT'],
                                    api_token=os.environ['NEPTUNE_API'],
                                    tags=self.tags,
                                    description=description)

        root_logger.addHandler(NeptuneHandler(run=self.run))


    @require_enabled
    def log_config(self, cfg):
        self.run['parameters'] = neptune.utils.stringify_unsupported(cfg)

    @require_enabled
    def log_metrics(self, metrics_dict, step=None):
        for key, value in metrics_dict.items():
            self.log_metric(key, value, step)

    @require_enabled
    def log_metric(self, metric, value, step=None):
        if step is not None:
            self.run[metric].append(value=value, step=step)
        else:
            self.run[metric].append(value)

    @require_enabled
    def log_results(self, metrics_dict):
        for key, value in metrics_dict.items():
            self.run[f'eval/{key}'] = value

    @require_enabled
    def __getitem__(self, item):
        return self.run.__getitem__(item)

    @require_enabled
    def __getattr__(self, item):
        return self.run.__getattribute__(item)
