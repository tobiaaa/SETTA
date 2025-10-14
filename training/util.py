import torch.optim.lr_scheduler as lr_sched


def get_lr_sched(cfg, opt):
    if cfg.name == 'exp':
        if cfg.gamma == 'auto':
            # Range is logarithmic [1,5e-4]
            gamma = 5e-4 ** (1 / cfg.epochs)
        else:
            gamma = cfg.gamma

        return lr_sched.ExponentialLR(opt, gamma) 
    else:
        raise ValueError(f'LR Scheduler "{cfg.name}" unkown')
