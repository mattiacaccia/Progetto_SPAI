import torch
import torch.nn as nn

class TODCNNWrapper(nn.Module):
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = nn.Sequential(
            # Primo blocco con stride 2 per ridurre la risoluzione a 320x320 e velocizzare drasticamente
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Secondo blocco con stride 2 per ridurre a 160x160 (velocità x4)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Convoluzioni dilatate sul feature map ridotto
            nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Head di predizione: 4 canali per il box + 1 per la confidenza dell'oggetto
        self.head = nn.Conv2d(128, 5 + num_classes, kernel_size=1)

    def forward(self, x, targets=None):
        features = self.backbone(x)
        raw_preds = self.head(features) # Shape: [B, 5, H_feat, W_feat]
        
        # CORRETTORE CRITICO PER LA mAP:
        # Applichiamo la Sigmoid ai primi 4 canali (box) per vincolarli tra 0 e 1, 
        # esattamente come i target normalizzati del dataset.
        box_preds = torch.sigmoid(raw_preds[:, :4, :, :])
        obj_preds = raw_preds[:, 4:, :, :] # Lasciamo i logits per la BCEWithLogitsLoss
        
        predictions = torch.cat([box_preds, obj_preds], dim=1)
        return predictions