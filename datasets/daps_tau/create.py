import logging
import os
import re
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import pyloudnorm as pln
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from util import ProgressIterator

logger = logging.getLogger(__name__)

MAX_SNR = 20.0
MIN_SNR = 0.0
RAMP_TIME_MS = 10.0
MIN_DURATION = 3.0
MAX_DURATION = 10.0
FS = 16_000


def get_index(cfg, data_cfg):
    if not os.path.exists(data_cfg.path):
        index = create_dataset(cfg, data_cfg)
    else:
        index = pd.read_csv(os.path.join(data_cfg.path, 'index.csv'))

    return index


def create_dataset(cfg, data_cfg):
    logger.info('Creating DAPS-TAU dataset')

    tau_path = Path(data_cfg.tau_path) / 'TAU-urban-acoustic-scenes-2019-development'
    daps_path = Path(data_cfg.daps_path) / 'clean'
    target_path = Path(data_cfg.path)

    if (tau_path / 'index.csv').exists():
        noise_index = pd.read_csv(tau_path / 'index.csv')
    else:
        noise_files = sorted(tau_path.rglob('*.wav'))
        noise_index = _index_files(noise_files)
        noise_index.to_csv(tau_path / 'index.csv', index=False)

    if (daps_path / 'index.csv').exists():
        speech_index = pd.read_csv(daps_path / 'index.csv')
    else:
        speech_files = sorted(daps_path.rglob('*.wav'))
        speech_files = filter(lambda x: re.match(r'^[mf].*wav', x.name), speech_files)
        speech_index = _index_files(speech_files)
        speech_index.to_csv(daps_path / 'index.csv', index=False)

    speech_index = _make_segments(speech_index)

    # Joint index
    gen = np.random.default_rng(1234)
    perm = gen.permutation(len(noise_index))[:len(speech_index)]
    snr = gen.uniform(MIN_SNR, MAX_SNR, len(speech_index))
    noise_index = noise_index.iloc[perm].reset_index(drop=True)
    noise_files = [Path(x).name for x in noise_index['path']]
    speech_index['noise_file'] = noise_files
    speech_index['snr'] = snr

    speech_index = create_premixed(speech_index, target_path, daps_path, tau_path / 'audio')

    speech_index.to_csv(target_path / 'index.csv', index=False)

    return speech_index


def create_premixed(index, target_path, clean_path, noise_path):
    ramp_samples = int(RAMP_TIME_MS * FS / 1000)
    ramp = np.linspace(0, 1, ramp_samples)

    target_path.mkdir(exist_ok=True)
    target_clean_path = target_path / 'clean'
    target_clean_path.mkdir(exist_ok=True)
    target_noisy_path = target_path / 'noisy'
    target_noisy_path.mkdir(exist_ok=True)

    meter = pln.Meter(FS)

    names = []
    keep = []

    for i, row in ProgressIterator(index.iterrows(), total=len(index)):
        target_name = f'{i:04d}.wav'

        clean_dec = AudioDecoder(clean_path / row['file'], sample_rate=FS)
        clean = clean_dec.get_samples_played_in_range(row['start'], row['end'])
        x_clean = clean.data
        clean_loudness = meter.integrated_loudness(x_clean.squeeze().numpy())

        noise_dec = AudioDecoder(noise_path / row['noise_file'], sample_rate=FS)
        noise = noise_dec.get_all_samples()
        x_noise = noise.data[0]
        noise_loudness = meter.integrated_loudness(x_noise.squeeze().numpy())

        if x_clean.shape != x_noise.shape:
            min_len = min(x_clean.shape[-1], x_noise.shape[-1])
            x_clean = x_clean[..., :min_len]
            x_noise = x_noise[..., :min_len]

        if not np.isfinite(clean_loudness) or not np.isfinite(noise_loudness):
            logger.warning('Loudness not defined')
            keep.append(False)
            continue
        else:
            keep.append(True)
            names.append(target_name)

        target_loudness = clean_loudness - row['snr']
        delta_loudness = target_loudness - noise_loudness
        gain = 10.0 ** (delta_loudness / 20.0)
        noise_scaled = gain * x_noise
        x_noisy = x_clean + noise_scaled

        # Apply ramp
        x_noisy[:, :ramp_samples] *= ramp
        x_noisy[:, -ramp_samples:] *= ramp[::-1]

        x_clean[:, :ramp_samples] *= ramp
        x_clean[:, -ramp_samples:] *= ramp[::-1]

        AudioEncoder(x_noisy, sample_rate=FS).to_file(target_noisy_path / target_name)
        AudioEncoder(x_clean, sample_rate=FS).to_file(target_clean_path / target_name)

    index = index[keep]
    index['mix_name'] = names

    return index


def _make_segments(index):
    segment_index = []
    for _, row in index.iterrows():
        num_segments = ceil(row['duration'] / MAX_DURATION)
        for segment in range(num_segments):
            start = segment * MAX_DURATION
            end = min((segment + 1) * MAX_DURATION, row['duration'])
            if end - start < MIN_DURATION:
                continue
            segment_index.append({
                'file': Path(row['path']).name,
                'start': start,
                'end': end,
                'start_frame': int(start * row['fs']),
                'end_frame': int(end * row['fs']),
                'fs': row['fs']
            })

    segment_index = pd.DataFrame(segment_index)
    return segment_index


def _index_files(paths):
    logger.info('Indexing...')
    index = []
    for path in ProgressIterator(paths):
        dec = AudioDecoder(path)
        meta = {'path': path,
                'duration': dec.metadata.duration_seconds,
                'fs': dec.metadata.sample_rate}
        index.append(meta)

    index = pd.DataFrame(index)
    return index
