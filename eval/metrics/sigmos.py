import logging
import os

import librosa
import numpy as np
import onnxruntime as ort
import requests
import scipy
import torch

from ._base import Metric

logger = logging.getLogger(__name__)

class SigMOS(Metric):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.resample_type = 'fft'

        # STFT params
        self.dft_size = 960
        self.frame_size = 480
        self.window_length = 960
        self.window = np.sqrt(np.hanning(int(self.window_length) + 1)[:-1]).astype(np.float32)

        model_path = self._load_model()

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, options)

    def update(self, x_clean, x_noisy, x_denoised, kwargs):
        x_denoised = x_denoised.squeeze().cpu().numpy()
        if self.fs != 48_000:
            x_denoised = librosa.resample(x_denoised, orig_sr=self.fs, target_sr=48_000, res_type=self.resample_type)

        features = self.stft(x_denoised)
        features = self.compressed_mag_complex(features)

        onnx_inputs = {inp.name: features for inp in self.session.get_inputs()}
        output = self.session.run(None, onnx_inputs)[0][0]
        output = torch.tensor(output)
        self.append(output)

        return output

    def _load_model(self):
        path = os.path.expanduser('~/.cache/speech_enhancement/sigmos/model.onnx')
        if not os.path.exists(path):
            # Download model
            logger.info('Downloading SigMOS model...')
            response = requests.get('https://github.com/microsoft/SIG-Challenge/raw/refs/heads/main/ICASSP2024/sigmos/model-sigmos_1697718653_41d092e8-epo-200.onnx')
            assert response.status_code == 200
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                f.write(response.content)

        return path        

    def names(self):
        return ['MOS_COL', 'MOS_DISC', 'MOS_LOUD', 'MOS_NOISE', 'MOS_REVERB', 'MOS_SIG', 'MOS_OVRL']

    def stft(self, signal):
        last_frame = len(signal) % self.frame_size
        if last_frame == 0:
            last_frame = self.frame_size

        padded_signal = np.pad(signal, ((self.window_length - self.frame_size, self.window_length - last_frame),))
        frames = librosa.util.frame(padded_signal, frame_length=len(self.window), hop_length=self.frame_size, axis=0)
        spec = scipy.fft.rfft(frames * self.window, n=self.dft_size)
        return spec.astype(np.complex64)

    @staticmethod
    def compressed_mag_complex(x: np.ndarray, compress_factor=0.3):
        x = x.view(np.float32).reshape(x.shape + (2,)).swapaxes(-1, -2)
        x2 = np.maximum((x * x).sum(axis=-2, keepdims=True), 1e-12)
        if compress_factor == 1:
            mag = np.sqrt(x2)
        else:
            x = np.power(x2, (compress_factor - 1) / 2) * x
            mag = np.power(x2, compress_factor / 2)

        features = np.concatenate((mag, x), axis=-2)
        features = np.transpose(features, (1, 0, 2))
        return np.expand_dims(features, 0)
