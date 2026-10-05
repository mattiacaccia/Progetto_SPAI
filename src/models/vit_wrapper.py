import torch
import torch.nn as nn
from transformers import AutoModelForObjectDetection

class ViTDetWrapper(nn.Module):
    def __init__(self, model_name: str = "facebook/detr-resnet-50", num_classes: int = 1):

        super().__init__()
        # Caricamento del modello Transformer da Hugging Face adattando il numero di classi
        self.model = AutoModelForObjectDetection.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, x,targets=None):
        return self.model(pixel_values=x)