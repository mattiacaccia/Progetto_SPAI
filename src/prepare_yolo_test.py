import json
import shutil
from pathlib import Path

# Crea la cartella YOLO con SOLO il test set isolato 

def generate_yolo_test_folder():
    project_root = Path(__file__).resolve().parent.parent
    base = project_root / "data" / "sliced_processed_FertyLab"
    imgs, lbls, splits_file = base / "images", base / "labels", base / "splits.json"

    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)
    mapping = splits["mapping_image_to_sample"]
    test_samples = set(splits["test_samples"])
    train_val_samples = set(splits["train_val_samples"])
    assert not (test_samples & train_val_samples), "Campioni sia in test che in train/val!"

    out = project_root / "data" / "yolo_test"
    out_imgs, out_lbls = out / "images", out / "labels"
    # ripulisce eventuali residui di run precedenti
    if out.exists():
        shutil.rmtree(out)
    out_imgs.mkdir(parents=True)
    out_lbls.mkdir(parents=True)

    n = 0
    for img_path in imgs.glob("*.*"):
        if img_path.name.startswith(".") or mapping.get(img_path.name) not in test_samples:
            continue
        shutil.copy(img_path, out_imgs / img_path.name)
        lbl = lbls / (img_path.stem + ".txt")
        if lbl.exists():
            shutil.copy(lbl, out_lbls / lbl.name)
        n += 1
    print(f"[TEST] {n} immagini di test isolate copiate in {out}")

    test_names = {p.name for p in out_imgs.iterdir()}
    for regime in ["yolo_regime_a", "yolo_regime_b"]:
        d = project_root / "data" / regime / "images"
        if d.exists():
            overlap = test_names & {p.name for p in d.iterdir()}
            print(f"[CHECK] {regime}: {len(overlap)} immagini di test presenti anche nel training (deve essere 0)")

if __name__ == "__main__":
    generate_yolo_test_folder()