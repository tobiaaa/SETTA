from .gan_trainer import GanTrainer
from .trainer import Trainer
from .emb import EmbTrainer


def get_trainer(cfg):
    if cfg.name in ('gan', 'metricgan'):
        return GanTrainer
    elif cfg.name == 'default':
        return Trainer
    elif cfg.name == 'emb':
        return EmbTrainer
    else:
        raise NameError(f'Training scheme "{cfg.name}" unkown')
