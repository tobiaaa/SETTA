from torchmetrics.text import WordErrorRate

from ._base import Metric


class WER(Metric):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.wer = WordErrorRate()

    def update(self, y_true, y_pred, kwargs):
        wer = self.wer(y_pred, y_true)
        self.append(wer)

        return wer
