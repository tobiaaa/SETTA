import importlib
import logging
import os
from pathlib import Path

_logger = logging.getLogger(__name__)

_import_warnings = []


def _replay_warnings():
    global _import_warnings
    for module, msg in _import_warnings:
        _logger.debug(f'Module "{module}" could not be imported: {msg}')

    _import_warnings = []


# Automatically import all modules and subpackages to discover adaptation methods
_base = os.path.dirname(__file__)
_modules = []
for entry in sorted(os.listdir(_base)):
    if entry.startswith(('_', '.')):
        continue
    path = os.path.join(_base, entry)
    if os.path.isdir(path):
        if not os.path.isfile(os.path.join(path, '__init__.py')):
            continue
        module = entry
    elif entry.endswith('.py'):
        module = Path(entry).stem
    else:
        continue

    try:
        _modules.append(importlib.import_module(f'.{module}', __name__))
    except ImportError as e:
        _import_warnings.append((module, e.msg))


def get_adaptation(cfg, model, recon, device):
    _replay_warnings()

    if 'adaptation' not in cfg:
        return
    if cfg.adaptation is None:
        return

    for mod in _modules:
        if hasattr(mod, cfg.adaptation.name):
            return getattr(mod, cfg.adaptation.name)(cfg.adaptation, model, recon, device)
    raise ValueError(f'Adaptation "{cfg.adaptation.name}" not found')
