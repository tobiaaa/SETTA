import torch
from torch import nn
import transformers


class Whisper(nn.Module):
    def __init__(self):
        super().__init__()
        self.proc= transformers.WhisperFeatureExtractor.from_pretrained('openai/whisper-large-v3',
                                                                        torch_dtype=torch.float16)
        self.model = transformers.WhisperModel.from_pretrained('openai/whisper-large-v3',
                                                               torch_dtype=torch.float16)

    def forward(self, x):
        feat = self.proc(x).input_features
        embs = self.model(feat)['encoder_hidden_states']
        print(embs.shape)
        return {'extract_features': embs}
