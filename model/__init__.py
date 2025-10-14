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

# Automatically import 'model.py' from all subdirectories to invoke ModelRegistry
_modules = next(os.walk(os.path.dirname(__file__)))[1]
_all = filter(lambda x: not x.startswith(('_', '.')),_modules)
for module in _all:
    try:
        importlib.import_module(f'.{module}.model', package='model')
    except ImportError as e:
        _logger.warning(f'Directory "{module}.model" could not be imported; Ignoring directory')
