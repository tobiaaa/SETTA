import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from .whisper import Whisper

from ..registry import ModelRegistry

@ModelRegistry.register
class Foundation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.model = _get_model(cfg.model)

    def forward(self, x):
        x = x.to(torch.float16).squeeze(1)
        output = self.model(x) 
        x_feat = output['extract_features'].mean(dim=1)
        x_feat = x_feat / x_feat.norm(dim=-1, keepdim=True)
        return x_feat


def _get_model(name):
    if name == 'WavLM':
        model = transformers.WavLMModel.from_pretrained("microsoft/wavlm-large",
                                                        torch_dtype=torch.float16)
    elif name == 'Wav2Vec':
        model = transformers.Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", 
                                                           torch_dtype=torch.float16)
    elif name == 'XLS-R':
        model = transformers.Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m", 
                                                           torch_dtype=torch.float16)
    elif name == 'XLS-RB':
        model = transformers.Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-2b", 
                                                           torch_dtype=torch.float16)
    elif name == 'Whisper':
        model = Whisper()
    else:
        raise NameError(f'Model "{name}" unknown')

    return model
