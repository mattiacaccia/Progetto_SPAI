import torch
import torch.nn as nn
from ultralytics import YOLO

class YOLOv8Wrapper(nn.Module):
    def __init__(self, model_name: str = "yolov8n.pt", num_classes: int = 1):

        super().__init__()
        # Caricamento del modello pretrained da Ultralytics
        self.yolo_model = YOLO(model_name)
        # Estrazione  dell'architettura PyTorch sottostante
        self.model = self.yolo_model.model

    def forward(self, x, targets=None):
        # Passaggio forward dell'immagine attraverso la rete
        return self.model(x)
    
    def train(self, mode: bool = True):
        self.training = mode
        try:
            super().train(mode)
        except TypeError:
            pass
        return self