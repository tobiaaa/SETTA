from .source import Source
from .remix_it import RemixIT
from .laden import LaDen


def get_adaptation(cfg, model, recon):
    if 'adaptation' not in cfg:
        return
    if cfg.adaptation is None:
        return

    if cfg.adaptation.name == 'Source':
        return Source(cfg.adaptation, model, recon)
    elif cfg.adaptation.name == 'LaDen':
        return LaDen(cfg.adaptation, model, recon)
    elif cfg.adaptation.name == 'RemixIT':
        return RemixIT(cfg.adaptation, model, recon)
    else:
        raise NameError(f'Adaptation method "{cfg.adaptation.name}" unkown')
