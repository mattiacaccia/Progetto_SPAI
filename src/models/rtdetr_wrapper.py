import torch
import torch.nn as nn
from ultralytics import RTDETR

class RTDETRWrapper(nn.Module):
    def __init__(self, model_name: str = "rtdetr-l.pt", num_classes: int = 1):

        super().__init__()
        self.rtdetr_model = RTDETR(model_name)
        self.model = self.rtdetr_model.model

    def forward(self, x, targets=None):
        return self.model(x)
    
    def train(self, mode: bool = True):
        self.training = mode
        try:
            super().train(mode)
        except TypeError:
            pass
        return self