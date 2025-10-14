import torch
import torchaudio.functional as AF
import whisper
from torchmetrics.text import WordErrorRate

from ._base import Metric


class WERSE(Metric):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.asr_model = whisper.load_model('tiny', device)
        self.options = whisper.DecodingOptions()
        self.wer_metric = WordErrorRate()        
        
        self.denoised_data = []
        self.noisy_data = []
        self.imp_data = []

    def update(self, x_clean, x_noisy, x_denoised, kwargs):
        if 'text' in kwargs.keys():
            clean_text = kwargs['text']
        else:
            clean_text = self._transcribe(x_clean)
        noisy_text = self._transcribe(x_noisy)
        denoised_text = self._transcribe(x_denoised)

        noisy_wer = self.wer_metric(noisy_text, clean_text) 
        denoised_wer = self.wer_metric(denoised_text, clean_text) 

        improvement = noisy_wer - denoised_wer
        
        result = torch.tensor([denoised_wer, noisy_wer, improvement])
        self.append(result)

        return result

    def _transcribe(self, x):
        if self.fs != 16_000:
            x = AF.resample(x, self.fs, 16_000)
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(x.squeeze(1)))
        output = self.asr_model.decode(mel, self.options)
        text = [out.text for out in output]

        return text

    def names(self):
        return ['WERSE Denoised', 'WERSE Noisy', 'WERSE Improvement']
