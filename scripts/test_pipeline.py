from pathlib import Path
import sys
import inspect

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.yolo_dataset import YoloFormatDataset
from src.models.yolov8_wrapper import YOLOv8Wrapper
from src.models.rtdetr_wrapper import RTDETRWrapper
from src.models.vit_wrapper import ViTDetWrapper
from src.models.tod_cnn_wrapper import TODCNNWrapper

def test_run():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Utilizzo dispositivo: {device}")

    images_dir = project_root / "data" / "processed_FertyLab" / "images"
    labels_dir = project_root / "data" / "processed_FertyLab" / "labels"

    # Trasformazione per ridimensionare le immagini a 640x640 e convertirle in tensori
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])

    print("Inizializzazione del Dataset...")
    dataset = YoloFormatDataset(images_dir=images_dir, labels_dir=labels_dir , transform=transform)
    print(f"Trovate {len(dataset)} immagini")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    
    images, targets = next(iter(dataloader))
    # Conversione della tupla di immagini in un unico tensore di batch [B, C, H, W]
    images_tensor = torch.stack(images).to(device)
    print(f"Batch caricato con successo. Shape del tensore batch: {images_tensor.shape}. Immagini nel batch: {len(images)}\n")

    print("Test Inizializzazione e Forward/Loss Pass Modelli...")
    
    models = [
        {"name": "ViTDet", "class": ViTDetWrapper, "kwargs": {"model_name": "facebook/detr-resnet-50", "num_classes": 1}},
        {"name": "YOLOv8", "class": YOLOv8Wrapper, "kwargs": {"model_name": "yolov8n.pt", "num_classes": 1}},
        {"name": "RTDETR", "class": RTDETRWrapper, "kwargs": {"model_name": "rtdetr-l.pt", "num_classes": 1}},
        {"name": "TODCNN", "class": TODCNNWrapper, "kwargs": {"num_classes": 1}}
    ]

    for config in models:
        name = config["name"]
        model_class = config["class"]
        parameters = config["kwargs"]

        print(f" Esecuzione test completo su {name}...")

        try:
            model = model_class(**parameters)
            model.to(device)
            # Test Forward Pass + Calcolo Loss
            model.train()
            # Verifica l'adattamento delle annotazioni
            outputs = model(images_tensor, targets)

            print(f"{name}: Inizializzazione e calcolo loss riusciti")

            # Pulizia memoria
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()    
                
        except Exception as e:
            print(f" Fallito test su {name}. Errore: {e}")

if __name__ == "__main__":
    test_run()