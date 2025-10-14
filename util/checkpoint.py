import logging
import os

import hydra
import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, cfg):
        self.checkpoint_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        self.enable = os.environ.get('TRAINING_RUN', default='False') == 'True'
        if not self.enable:
            logger.info('Not Saving checkpoints. Run "export TRAINING_RUN=True"')

        self.freq = cfg.checkpoint_freq

    def save_model(self, state_dict, epoch=None, overwrite=True):
        if not self.enable:
            return
        if epoch is not None:
            filename = f'model_{epoch}.th'
        else:
            filename = 'model.th'

        if not overwrite:
            path = os.path.join(self.checkpoint_dir, filename)
            if os.path.exists(path):
                filename = self._alt_name(filename)

        torch.save(state_dict, os.path.join(self.checkpoint_dir, filename))

    def save_results(self, results, epoch=None, overwrite=True):
        if not self.enable:
            return
        if epoch is not None:
            filename = f'results_{epoch}.txt'
        else:
            filename = 'results.txt'

        if not overwrite:
            path = os.path.join(self.checkpoint_dir, filename)
            if os.path.exists(path):
                filename = self._alt_name(filename)

        with open(os.path.join(self.checkpoint_dir, filename), 'w') as f:
            f.write(results)

    def _alt_name(self, filename, max_iter=64):
        for i in range(max_iter):
            new_filename = f'{filename}%{i}'
            if os.path.exists(os.path.join(self.checkpoint_dir, new_filename)):
                continue
            return new_filename

        raise FileExistsError
