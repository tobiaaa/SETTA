import logging
import os
import sys
import time
from itertools import dropwhile

from tqdm import tqdm

logger = logging.getLogger(__name__)


class ProgressIterator:
    def __init__(self,
                 iterator=None,
                 desc=None,
                 total=None,
                 disable=False,
                 initial=0,
                 **kwargs):
        self._interactive = sys.stdin.isatty()
        if os.environ.get('SE_FORCE_TQDM', 'False') == 'True':
            logger.debug('Forcing TQDM')
            self._interactive = True
        elif os.environ.get('SE_DISABLE_TQDM', 'False') == 'True':
            logger.debug('Disabling TQDM')
            self._interactive = False

        self._iterator = iterator
        self._desc = desc
        self._total = total
        self._disable = disable
        self.initial = initial
        self._kwargs = kwargs

        self._iterable = None

        if not self._interactive and iterator is None:
            iter(self)

    def __iter__(self):
        if self._interactive:
            self._iterable = tqdm(self._iterator,
                                  desc=self._desc,
                                  total=self._total,
                                  disable=self._disable,
                                  initial=self.initial,
                                  **self._kwargs)

        else:
            self._iterable = _ProgressLogger(self._iterator,
                                             desc=self._desc,
                                             total=self._total,
                                             disable=self._disable,
                                             initial=self.initial)

        return iter(self._iterable)

    def set_postfix(self, postfix):
        self._iterable.set_postfix(postfix)

    def update(self, step):
        if self._iterable is None:
            iter(self)
        self._iterable.update(step)


class _ProgressLogger:
    def __init__(self, iterator=None, desc=None, total=None, disable=False, initial=0):
        self._iterator = iterator
        self._desc = desc
        self._total = total
        self._disable = disable

        if self._total is None:
            if hasattr(self._iterator, '__len__'):
                self._total = len(self._iterator)

        self._iterable = None

        self._initial = initial
        self._i = initial

        name = 'Prog'
        if desc is not None:
            name = f'{name}.{desc}'

        self._info_logger = logging.getLogger(name + '.Info')
        self._time_logger = logging.getLogger(name + '.Time')

        self._info_freq = int(os.environ.get('SE_PROG_INFO_F', '50'))
        self._time_freq = int(os.environ.get('SE_PROG_TIME_F', '50'))

        self._time_alpha = float(os.environ.get('SE_PROG_TIME_ALPHA', '0.8'))
        self._step_time = None
        self._last_time = None
        self._start_time = None
        self._last_post = {}

    def __iter__(self):
        if hasattr(self._iterator, '__iter__'):
            self._iterable = iter(self._iterator)
        now = time.perf_counter()
        self._last_time = now
        self._start_time = now
        self._i = self._initial
        return self

    def __next__(self):
        now = time.perf_counter()
        step_time = now - self._last_time
        if self._step_time is None:
            self._step_time = step_time

        self._step_time = self._time_alpha * self._step_time + (1.0 - self._time_alpha) * step_time

        if self._i % self._time_freq == 0 and not self._disable:
            self._log_time()

        if self._i % self._info_freq == 0 and not self._disable:
            self._log_info()

        self._i += 1
        self._last_time = now
        return next(self._iterable)

    def set_postfix(self, postfix):
        self._last_post.update(postfix)

    def update(self, diff):
        step = self._i + diff

        now = time.perf_counter()
        step_time = (now - self._last_time) / diff
        if self._step_time is None:
            self._step_time = step_time

        alpha = self._time_alpha ** diff
        self._step_time = alpha * self._step_time + (1.0 - alpha) * step_time

        if step // self._time_freq != self._i // self._time_freq and not self._disable:
            self._log_time()

        if self._i % self._info_freq == 0 and not self._disable:
            self._log_info()

        self._i = step
        self._last_time = now

    def _log_info(self):
        step_str = f'Step {self._i}/{self._total}'
        output = []
        for key, val in self._last_post.items():
            if type(val) in [float, int]:
                output.append(f'{key}: {val:.4g}')
            else:
                output.append(f'{key}: {val}')
        if output:
            post_str = ' -> ' + '\t'.join(output)
            self._info_logger.info(step_str + post_str)
        else:
            self._info_logger.debug('No info logged')

    def _log_time(self):
        now = time.perf_counter()
        elapsed = now - self._start_time

        if self._total is not None:
            total = elapsed + self._step_time * (self._total - self._i)
            step_str = f'Step {self._i}/{self._total}'
            time_str = f'{self._format_time(elapsed)}/{self._format_time(total)}'
            self._time_logger.info(f'{step_str} -> {time_str}')

        else:
            self._time_logger.info(f'{self._format_time(elapsed)}')

    def _format_time(self, time):
        time = round(time)
        days, time = divmod(time, 24 * 60 * 60)
        hours, time = divmod(time, 60 * 60)
        minutes, seconds = divmod(time, 60)
        time = [hours, minutes, seconds]
        time = [f'{x:02d}' for x in time]
        if days:
            output = f'{days}-{":".join(time)}'
        else:
            if int(hours) == 0:
                time = time[1:]
            output = ':'.join(time)

        return output
