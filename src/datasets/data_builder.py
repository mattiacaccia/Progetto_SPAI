import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import transforms
from src.datasets.yolo_dataset import YoloFormatDataset

# Split anti-leakage: la validation è un CAMPIONE intero (mai tile dello stesso campione nel training).
DEFAULT_VAL_SAMPLES = ("SEM-0004",)


def _collate(x):
    return tuple(zip(*x))


def get_train_val_dataloaders(regime: str = "regime_b", batch_size: int = 8, img_size: int = 640,
                              val_samples=DEFAULT_VAL_SAMPLES, num_workers: int = 0):
    project_root = Path(__file__).resolve().parent.parent.parent

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    ferty_imgs = project_root / "data" / "sliced_processed_FertyLab" / "images"
    ferty_lbls = project_root / "data" / "sliced_processed_FertyLab" / "labels"
    splits_file = project_root / "data" / "sliced_processed_FertyLab" / "splits.json"

    svia_imgs = project_root / "data" / "sliced_processed_SVIA" / "images"
    svia_lbls = project_root / "data" / "sliced_processed_SVIA" / "labels"

    if not splits_file.exists():
        raise FileNotFoundError("File splits.json non trovato")
    with open(splits_file, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    mapping = splits_data["mapping_image_to_sample"]
    test_samples = set(splits_data["test_samples"])
    train_val_samples = set(splits_data["train_val_samples"])

    # ---- TEST SET (campioni isolati) ----
    if regime == "test":
        ferty_dataset = YoloFormatDataset(ferty_imgs, ferty_lbls, transform=transform)
        test_indices = [
            idx for idx, img_name in enumerate(ferty_dataset.image_files)
            if mapping.get(img_name) in test_samples
        ]
        test_subset = Subset(ferty_dataset, test_indices)
        print(f"DataLoader Test caricato con successo: {len(test_subset)} immagini di test rigorosamente isolate")
        return DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=_collate, num_workers=num_workers)

    if regime not in ("regime_a", "regime_b"):
        raise ValueError(f"Regime sconosciuto: '{regime}'. Scegli tra 'regime_a', 'regime_b' o 'test'")

    # ---- TRAIN / VALIDATION divisi PER CAMPIONE ----
    val_s = set(splits_data.get("val_samples", val_samples))
    if not val_s <= train_val_samples:
        raise ValueError(f"val_samples {sorted(val_s)} non è contenuto in train_val_samples {sorted(train_val_samples)}")
    train_s = train_val_samples - val_s
    if not train_s or not val_s:
        raise ValueError("Split non valido: servono almeno 1 campione di training e 1 di validation")
    if (train_s | val_s) & test_samples:
        raise ValueError("Leakage: campioni di test presenti in train/val!")

    ferty_dataset = YoloFormatDataset(ferty_imgs, ferty_lbls, transform=transform)
    train_indices = [i for i, n in enumerate(ferty_dataset.image_files) if mapping.get(n) in train_s]
    val_indices = [i for i, n in enumerate(ferty_dataset.image_files) if mapping.get(n) in val_s]
    if len(val_indices) == 0 or len(train_indices) == 0:
        raise RuntimeError(f"Split vuoto: train={len(train_indices)} val={len(val_indices)} tile. Controlla splits.json")

    train_dataset = Subset(ferty_dataset, train_indices)
    val_dataset = Subset(ferty_dataset, val_indices)
    print(f"[SPLIT {regime}] train: {sorted(train_s)} ({len(train_indices)} tile) | "
          f"val: {sorted(val_s)} ({len(val_indices)} tile) | test escluso: {sorted(test_samples)}")

    if regime == "regime_a":
        svia_dataset = YoloFormatDataset(svia_imgs, svia_lbls, transform=transform)
        train_dataset = ConcatDataset([train_dataset, svia_dataset])
        print(f"[SPLIT regime_a] + SVIA nel training: {len(svia_dataset)} immagini")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate, num_workers=num_workers)
    return train_loader, val_loader


def get_dataloaders(regime: str = "regime_b", batch_size: int = 8, img_size: int = 640):
    """Compatibilità: test.py usa get_dataloaders(regime='test')."""
    if regime == "test":
        return get_train_val_dataloaders(regime="test", batch_size=batch_size, img_size=img_size)
    train_loader, _ = get_train_val_dataloaders(regime=regime, batch_size=batch_size, img_size=img_size)
    return train_loader