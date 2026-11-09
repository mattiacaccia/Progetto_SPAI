
import argparse
import json
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.datasets.data_builder import DEFAULT_VAL_SAMPLES  # stesso split usato da train_vit.py e train_todcnn.py

IMG_EXT = {".jpg", ".jpeg", ".png"}


def list_images(folder: Path):
    return sorted(p for p in folder.glob("*.*") if not p.name.startswith(".") and p.suffix.lower() in IMG_EXT)


def copy_pair(img: Path, lbl_dir: Path, out_img_dir: Path, out_lbl_dir: Path):
    shutil.copy2(img, out_img_dir / img.name)
    lbl = lbl_dir / (img.stem + ".txt")
    if lbl.exists():
        shutil.copy2(lbl, out_lbl_dir / lbl.name)
    else:  # nessuna label = sfondo
        (out_lbl_dir / (img.stem + ".txt")).write_text("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Cancella e ricrea data/yolo_regime_a e data/yolo_regime_b")
    args = ap.parse_args()

    data = project_root / "data"
    ferty_imgs, ferty_lbls = data / "sliced_processed_FertyLab" / "images", data / "sliced_processed_FertyLab" / "labels"
    svia_imgs, svia_lbls = data / "sliced_processed_SVIA" / "images", data / "sliced_processed_SVIA" / "labels"
    splits_file = data / "sliced_processed_FertyLab" / "splits.json"
    if not splits_file.exists():
        raise FileNotFoundError(f"{splits_file} non trovato")

    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)
    mapping = splits["mapping_image_to_sample"]
    test_s = set(splits["test_samples"])
    trainval_s = set(splits["train_val_samples"])
    val_s = set(splits.get("val_samples", DEFAULT_VAL_SAMPLES))
    if not val_s <= trainval_s:
        raise ValueError(f"val_samples {sorted(val_s)} non contenuto in train_val_samples {sorted(trainval_s)}")
    train_s = trainval_s - val_s
    print(f"[SPLIT] train: {sorted(train_s)} | val: {sorted(val_s)} | test (escluso): {sorted(test_s)}")

    dests = {r: data / f"yolo_{r}" for r in ("regime_a", "regime_b")}
    existing = [d for d in dests.values() if d.exists()]
    if existing and not args.force:
        print("Esistono già:", *[str(d) for d in existing], sep="\n  ")
        print("Rilancia con --force per rigenerarle (vengono cancellate e ricreate: sono copie derivate, i dati originali non si toccano).")
        sys.exit(1)
    for d in existing:
        shutil.rmtree(d)

    ferty = list_images(ferty_imgs)
    names_by_split = {"train": set(), "val": set()}
    for regime, dest in dests.items():
        for sub in ("images/train", "images/val", "labels/train", "labels/val"):
            (dest / sub).mkdir(parents=True)
        counts = {"train": 0, "val": 0, "svia": 0}
        for img in ferty:
            sample = mapping.get(img.name)
            split = "train" if sample in train_s else "val" if sample in val_s else None
            if split is None:
                continue  # test o non mappato
            copy_pair(img, ferty_lbls, dest / "images" / split, dest / "labels" / split)
            counts[split] += 1
            names_by_split[split].add(img.name)
        if regime == "regime_a":
            for img in list_images(svia_imgs):
                copy_pair(img, svia_lbls, dest / "images" / "train", dest / "labels" / "train")
                counts["svia"] += 1
        print(f"[{regime.upper()}] train: {counts['train'] + counts['svia']} tile "
              f"(FertyLab {counts['train']}" + (f" + SVIA {counts['svia']}" if regime == 'regime_a' else "") +
              f") | val: {counts['val']} tile (FertyLab)")

    # controlli anti-leakage
    test_names = {n for n, s in mapping.items() if s in test_s}
    for regime, dest in dests.items():
        present = {p.name for sub in ("train", "val") for p in (dest / "images" / sub).iterdir()}
        leak = present & test_names
        overlap = {p.name for p in (dest / "images" / "train").iterdir()} & {p.name for p in (dest / "images" / "val").iterdir()}
        print(f"[CHECK {regime}] tile di test presenti: {len(leak)} | tile presenti sia in train che in val: {len(overlap)} (entrambi devono essere 0)")
        if leak or overlap:
            raise RuntimeError("Leakage rilevato!")


if __name__ == "__main__":
    main()