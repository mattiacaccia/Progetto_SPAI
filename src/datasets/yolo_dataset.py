import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class YoloFormatDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):

        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        
        # Ricerca tutte le immagini nella cartella
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Caricamento immagine
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # Ricerca txt corrispondente
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_name)
        
        boxes = []
        labels = []

        # Lettura del txt 
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        # Coordinate YOLO
                        cx, cy, w, h = map(float, parts[1:])
                        boxes.append([cx, cy, w, h])
                        labels.append(class_id)
        
        # Converte tutto in Tensori PyTorch per la GPU
        boxes = torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.int64) if len(labels) > 0 else torch.zeros((0,), dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels
        }

        # Applica eventuali trasformazioni 
        if self.transform:
            image = self.transform(image)
            
        return image, target