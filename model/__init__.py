import logging
import os
import warnings
import importlib

from . import util
from .registry import ModelRegistry
from .gan_wrapper import GANWrapper
from .emb_approx import EmbeddingApproximator

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

_logger = logging.getLogger(__name__)

# Automatically import subpackages (or 'model.py' from plain subdirectories) to invoke ModelRegistry
_base = os.path.dirname(__file__)
_modules = next(os.walk(_base))[1]
_all = filter(lambda x: not x.startswith(('_', '.')), _modules)
for module in _all:
    if os.path.isfile(os.path.join(_base, module, '__init__.py')):
        target = f'.{module}'
    else:
        target = f'.{module}.model'
    try:
        importlib.import_module(target, package='model')
    except ImportError as e:
        util._import_warnings.append((module, e.msg))
