import logging
from abc import ABC, abstractmethod


class LogHelper(ABC):
    def __init__(self, cfg, cfg_tot):
        self.freq = cfg.freq

        # Tags
        self.tags = [cfg_tot.model.name]
        if hasattr(cfg_tot.data, 'testset'):
            self.tags += [cfg_tot.data.trainset.name, cfg_tot.data.testset.name]
            if cfg_tot.data.trainset.name != cfg_tot.data.testset.name:
                self.tags.append('Domain Shift')

            if cfg_tot.data.trainset.name == 'CommonVoice':
                self.tags.append(cfg_tot.data.trainset.language)
            if cfg_tot.data.testset.name == 'CommonVoice':
                self.tags.append(cfg_tot.data.testset.language)

            if cfg_tot.data.trainset.name == 'DNS':
                self.tags += list(cfg_tot.data.trainset.languages)
            if cfg_tot.data.testset.name == 'DNS':
                self.tags += list(cfg_tot.data.testset.languages)

        else:
            self.tags.append(cfg_tot.data.dataset.name)

            if cfg_tot.data.dataset.name == 'CommonVoice':
                self.tags.append(cfg_tot.data.dataset.language)

            if cfg_tot.data.dataset.name == 'DNS':
                self.tags += list(cfg_tot.data.dataset.languages)

        if hasattr(cfg_tot.data, "reverb"):
            self.tags.append('Reverb')

    def log_config(self, cfg):
        return

    @abstractmethod
    def log_metrics(self, metrics_dict, step=None):
        self.run.log(metrics_dict, step)

    @abstractmethod
    def log_results(self, metrics_dict):
        self.log_metrics(metrics_dict)

    def log_class_code(self, method):
        logging.warning(f'Class logging not implemented for {self.__class__} logger')

def require_enabled(func):
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'enabled') and self.enabled:
            return func(self, *args, **kwargs)
    return wrapper
