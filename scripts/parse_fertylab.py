import json
import shutil
from pathlib import Path

def process_fertylab_from_yolo(only_sperm_class: bool = False):

    project_root = Path(__file__).resolve().parent.parent

    raw_fertylab_dir = project_root / "data" / "DataSet_FertyLab"
    raw_labels_dir = raw_fertylab_dir / "yolo" / "labels"
    raw_images_dir = raw_fertylab_dir / "images"  

    if not raw_images_dir.exists():
        raw_images_dir = raw_fertylab_dir

    output_dir = project_root / "data" / "processed_FertyLab"
    output_imgs = output_dir / "images"
    output_lbls = output_dir / "labels"

    output_imgs.mkdir(parents=True, exist_ok=True)
    output_lbls.mkdir(parents=True, exist_ok=True)

    txt_files = list(raw_labels_dir.glob("*.txt"))
    print(f"Trovati {len(txt_files)} file di annotazione YOLO in FertyLab.")

    samples_map = {}
    count = 0

    for txt_path in txt_files:
        stem = txt_path.stem  
        
        # Estrazione del Sample ID dal nome del file 
        sample_id = stem.split("_")[0]
        
        # Cerca l'immagine corrispondente 
        img_path = raw_images_dir / f"{stem}.jpg"
        if not img_path.exists():
            img_path = raw_images_dir / f"{stem}.png"

        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if only_sperm_class:
            # Mantieniamo solo le annotazioni con class_id == 0 (TESTA_SPERMATOZOO)
            lines = [line for line in lines if line.startswith("0 ")]

        out_lbl_path = output_lbls / f"{stem}.txt"
        with open(out_lbl_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        if img_path.exists():
            shutil.copy2(img_path, output_imgs / img_path.name)
            samples_map[img_path.name] = sample_id

        count += 1

    print(f"Completata la strutturazione di {count} frame in {output_dir}")

    # Creazione della partizione per sample_id
    all_samples = list(set(samples_map.values()))
    
    # Test Set fisso su 2 campioni specifici 
    test_samples = [s for s in ['SEM-0002', 'SEM-0013'] if s in all_samples]
    train_val_samples = [s for s in all_samples if s not in test_samples]

    splits = {
        'all_samples': all_samples,
        'train_val_samples': train_val_samples,
        'test_samples': test_samples,
        'mapping_image_to_sample': samples_map
    }

    with open(output_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=4)

    print(f"File splits.json generato correttamente.")
    print(f"Sample per Train/Val: {train_val_samples}")
    print(f"Sample riservati ESCLUSIVAMENTE al Test Set: {test_samples}")

if __name__ == "__main__":
    # only_sperm_class=True per addestrare i modelli SOLO sugli spermatozoi
    process_fertylab_from_yolo(only_sperm_class=True)