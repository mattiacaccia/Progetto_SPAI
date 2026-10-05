import torch
import torch.nn as nn

class TODCNNWrapper(nn.Module):

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = nn.Sequential(
            # estrazione ad alta risoluzione 
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # convoluzione dilatata r=2 per catturare il contesto vicino
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # convoluzione dilatata r=4 per espandere il campo ricettivo globale
            nn.Conv2d(64, 128, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # raffinamento delle feature ad alta densità
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Canali output = 5 (bbox + conf) + num_classes
        self.head = nn.Conv2d(128, 5 + num_classes, kernel_size=1)

    def forward(self, x, targets=None):

        features = self.backbone(x)
        predictions = self.head(features)
        return predictions