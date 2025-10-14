import logging
import os
from functools import partial

import numpy as np
import pandas as pd
import pyloudnorm as pln
import torchaudio
import torchaudio.functional as AF
from tqdm.contrib.concurrent import process_map

logger = logging.getLogger(__name__)

FS = 16_000
TEST_DIRS = ['SPSQUARE', 'PCAFETER', 'OMEETING', 'OHALLWAY']

def get_index(cfg, split, noise_dir):
    top_dir = cfg.path
    
    noise_df = get_noise_index(cfg, noise_dir)
    data_df_path = os.path.join(top_dir, f'{split}_index.csv')
    if os.path.exists(data_df_path):
        data_df = pd.read_csv(data_df_path)
        return data_df, noise_df

    logger.warning('Index not found. Creating index')
    np.random.seed(123)

    data_dir = os.path.join(top_dir, split, 'clean')

    dirs = [dir for dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, dir))]
    files = [[(dir, file) for file in os.listdir(os.path.join(data_dir, dir))] for dir in dirs]
    files = sum(files, [])

    output = process_map(partial(index_worker, top_dir=data_dir),
                         files,
                         total=len(files),
                         desc='Creating data index')

    data_df = pd.DataFrame(output)

    n_sample = len(data_df)
    snr = np.random.rand(n_sample) * 20 - 2.5
    data_df['SNR'] = snr

    n_test = len(noise_df[noise_df['split'] == split])
    min_test = noise_df[noise_df['split'] == split].index[0]
    noise_idx = (np.random.rand(n_sample) * n_test + min_test).astype(int) 

    data_df['noise_idx'] = noise_idx

    data_df.to_csv(data_df_path, index=False)

    return data_df, noise_df


def index_worker(row, top_dir):
    path = os.path.join(top_dir, *row)

    audio, fs = torchaudio.load(path)
    if fs != FS:
        audio = AF.resample(audio, fs, FS) 


    meter = pln.Meter(FS)
    loudness = meter.integrated_loudness(audio[0].numpy())
    length = audio.shape[0] / FS
    
    output = {
        'filename': path,
        'loudness': loudness,
        'duration': length
    }
    
    return output


def get_noise_index(cfg, top_dir):
    index_path = os.path.join(top_dir, 'index.csv')
    if os.path.exists(index_path):
        return pd.read_csv(index_path)

    logger.warning('Noise index not found. Creating index')
    np.random.seed(123)

    dirs = os.listdir(top_dir)

    files = [[(dir, file) for file in os.listdir(os.path.join(top_dir, dir))] for dir in dirs]
    files = sum(files, [])

    index = process_map(partial(_noise_index_worker, top_dir=top_dir),
                        files,
                        total=len(files),
                        max_workers=4,
                        desc='Creating noise index')

    index = sum(index, start=[])
    index_df = pd.DataFrame(index)

    dirs = index_df['dir'].isin(TEST_DIRS)
    split = dirs.map(lambda x: 'test' if x else 'train')
    index_df['split'] = split
    
    index_df.to_csv(index_path, index=False)

    return index_df


def _noise_index_worker(row, top_dir, segment_length=30.0):
    path = os.path.join(top_dir, *row)

    audio, fs = torchaudio.load(path)
    if fs != FS:
        audio = AF.resample(audio, fs, FS)

    length = audio.shape[1]
    length_seconds = length / FS
    n_segments = int(length_seconds / segment_length)
    
    meter = pln.Meter(FS)
    audio = audio[0, :int(n_segments*segment_length*FS)]
    # Segments
    segments = audio.reshape((-1, int(segment_length*FS)))
    loudness = [meter.integrated_loudness(seg) for seg in segments.numpy()]
    starts = [int(segment_length * FS * i) for i in range(n_segments)]
    output = [{'file': row[1],
               'dir': row[0],
               'start': start,
               'loudness': loudn} for start, loudn in zip(starts, loudness)]

    return output
