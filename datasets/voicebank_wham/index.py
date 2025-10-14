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

SPLIT_DICT = {
    'train': 'clean_trainset_28spk_wav',
    'test': 'clean_testset_wav'
}

def get_index(cfg, split, noise_dir):
    top_dir = cfg.path
    
    noise_df = get_noise_index(cfg)
    data_df_path = os.path.join(top_dir, f'{split}_index.csv')
    if os.path.exists(data_df_path):
        data_df = pd.read_csv(data_df_path)
        return data_df, noise_df

    logger.warning('Index not found. Creating index')
    np.random.seed(123)

    data_dir = os.path.join(top_dir, SPLIT_DICT[split])

    files = os.listdir(data_dir)

    output = process_map(partial(index_worker, top_dir=data_dir),
                         files,
                         total=len(files),
                         desc='Creating data index')

    data_df = pd.DataFrame(output)

    n_sample = len(data_df)
    snr = np.random.rand(n_sample) * 20
    if split == 'train':
        # [-2.5, 17.5] for train; [0,20] for testing
        snr -= 2.5
    data_df['SNR'] = snr

    n_test = len(noise_df[noise_df['split'] == split.capitalize()])
    min_test = noise_df[noise_df['split'] == split.capitalize()].index[0]
    noise_idx = (np.random.rand(n_sample) * n_test + min_test).astype(int) 

    data_df['noise_idx'] = noise_idx

    data_df.to_csv(data_df_path, index=False)

    return data_df, noise_df


def index_worker(row, top_dir):
    path = os.path.join(top_dir, row)

    audio, fs = torchaudio.load(path)
    if fs != FS:
        audio = AF.resample(audio, fs, FS) 

    meter = pln.Meter(FS)
    loudness = meter.integrated_loudness(audio[0].numpy())
    length = audio.shape[1] / FS
    
    output = {
        'filename': path,
        'loudness': loudness,
        'duration': length
    }
    
    return output


def get_noise_index(cfg):
    top_dir = os.path.join(cfg.noise_path, 'high_res_wham')
    index_path = os.path.join(top_dir, 'index.csv')
    if os.path.exists(index_path):
        return pd.read_csv(index_path)

    logger.warning('Noise index not found. Creating index')

    meta_df = pd.read_csv(os.path.join(top_dir, 'high_res_metadata.csv'))
    index = []
    index = process_map(partial(_noise_index_worker, top_dir=os.path.join(top_dir, 'audio')),
                        meta_df.iterrows(),
                        total=len(meta_df),
                        max_workers=4,
                        desc='Creating noise index')
    index = sum(index, start=[])
    index_df = pd.DataFrame(index)
    index_df.to_csv(index_path, index=False)

    return index_df


def _noise_index_worker(row, top_dir, segment_length=10.0):
    if isinstance(row, tuple):
        row = row[1]
    filename = row['Filename']
    duration = row['File Length (sec)']
    split = row['WHAM! Split']
    n_segments = int(duration / segment_length)
    if n_segments == 0:
        return []

    audio, fs = torchaudio.load(os.path.join(top_dir, filename))
    if fs != FS:
        audio = AF.resample(audio, fs, FS)
    
    meter = pln.Meter(FS)
    # Truncate and use channel 2
    audio = audio[1, :int(n_segments*segment_length*FS)]
    # Segments
    segments = audio.reshape((-1, int(segment_length*FS)))
    loudness = [meter.integrated_loudness(seg) for seg in segments.numpy()]
    starts = [int(segment_length * FS * i) for i in range(n_segments)]
    output = [{'file': filename,
               'split': split,
               'start': start,
               'loudness': loudn} for start, loudn in zip(starts, loudness)]

    return output
