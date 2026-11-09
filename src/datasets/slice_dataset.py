import os
from pathlib import Path
from PIL import Image
import json

def slice_image_and_labels(img_path, lbl_path, output_imgs_dir, output_lbls_dir, tile_size=640, overlap=128):
    if not img_path.exists() or img_path.name.startswith('.'):
        return

    try:
        img = Image.open(img_path)
    except Exception:
        return

    img_w, img_h = img.size

    # Leggi le label YOLO esistenti (formato: class xc yc w h normalizzati)
    bboxes = []
    if lbl_path.exists():
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
                
                # Converti da normalizzato a coordinate assolute in pixel
                xmin = (xc - w / 2) * img_w
                ymin = (yc - h / 2) * img_h
                xmax = (xc + w / 2) * img_w
                ymax = (yc + h / 2) * img_h
                bboxes.append((cls, xmin, ymin, xmax, ymax))

    stride = tile_size - overlap
    
    # Calcola le patch scorrevoli
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            # Gestione bordi dell'immagine
            x_end = min(x + tile_size, img_w)
            y_end = min(y + tile_size, img_h)
            x_start = max(0, x_end - tile_size)
            y_start = max(0, y_end - tile_size)
            
            tile_w = x_end - x_start
            tile_h = y_end - y_start

            # Crop dell'immagine
            tile_img = img.crop((x_start, y_start, x_end, y_end))

            # Se la patch ai bordi è più piccola del tile_size, facciamo il resize esatto
            if tile_w != tile_size or tile_h != tile_size:
                tile_img = tile_img.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
                scale_x = tile_size / tile_w
                scale_y = tile_size / tile_h
            else:
                scale_x = scale_y = 1.0

            tile_bboxes = []
            for cls, xmin, ymin, xmax, ymax in bboxes:
                # Controlla se la bounding box interseca la patch corrente
                if xmax > x_start and xmin < x_end and ymax > y_start and ymin < y_end:
                    # Ricalcola coordinate relative alla patch
                    bx1 = max(xmin, x_start) - x_start
                    by1 = max(ymin, y_start) - y_start
                    bx2 = min(xmax, x_end) - x_start
                    by2 = min(ymax, y_end) - y_start

                    # Applica eventuale riscalatura del crop ai bordi
                    bx1 *= scale_x
                    by1 *= scale_y
                    bx2 *= scale_x
                    by2 *= scale_y

                    # Riconverti in formato YOLO normalizzato rispetto al tile_size
                    bw = bx2 - bx1
                    bh = by2 - by1
                    bxc = bx1 + bw / 2
                    byc = by1 + bh / 2

                    if bw > 0 and bh > 0:
                        nx_c = bxc / tile_size
                        ny_c = byc / tile_size
                        nw = bw / tile_size
                        nh = bh / tile_size
                        tile_bboxes.append(f"{cls} {nx_c:.6f} {ny_c:.6f} {nw:.6f} {nh:.6f}\n")

            # Salva la patch e le relative label solo se contiene almeno uno spermatozoo
            if tile_bboxes:
                tile_name = f"{img_path.stem}_tile_{x_start}_{y_start}"
                tile_img.save(output_imgs_dir / f"{tile_name}.jpg", quality=95)
                with open(output_lbls_dir / f"{tile_name}.txt", "w", encoding="utf-8") as f:
                    f.writelines(tile_bboxes)

def process_dataset_slicing():
    project_root = Path(__file__).resolve().parent.parent.parent
    
    for dataset_name in ["processed_FertyLab", "processed_SVIA"]:
        src_imgs = project_root / "data" / dataset_name / "images"
        src_lbls = project_root / "data" / dataset_name / "labels"
        
        out_dir = project_root / "data" / f"sliced_{dataset_name}"
        out_imgs = out_dir / "images"
        out_lbls = out_dir / "labels"
        out_imgs.mkdir(parents=True, exist_ok=True)
        out_lbls.mkdir(parents=True, exist_ok=True)

        print(f"Avvio slicing per {dataset_name}...")
        
        # Filtro rigoroso: ignora file nascosti e prende solo estensioni di immagini valide
        image_files = [
            f for f in src_imgs.glob("*.*") 
            if not f.name.startswith('.') and f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ]
        
        for img_path in image_files:
            lbl_path = src_lbls / f"{img_path.stem}.txt"
            slice_image_and_labels(img_path, lbl_path, out_imgs, out_lbls, tile_size=640, overlap=128)
            
        print(f"Slicing completato per {dataset_name}. Salvato in {out_dir}")

        # Aggiornamento del file splits.json per FertyLab
        if dataset_name == "processed_FertyLab":
            old_splits = src_imgs.parent / "splits.json"
            if old_splits.exists():
                with open(old_splits, "r", encoding="utf-8") as f:
                    splits_data = json.load(f)
                
                old_mapping = splits_data["mapping_image_to_sample"]
                new_mapping = {}
                for img_name, sample_id in old_mapping.items():
                    stem = Path(img_name).stem
                    for sliced_img in out_imgs.glob(f"{stem}_tile_*.jpg"):
                        new_mapping[sliced_img.name] = sample_id
                
                splits_data["mapping_image_to_sample"] = new_mapping
                with open(out_dir / "splits.json", "w", encoding="utf-8") as f:
                    json.dump(splits_data, f, indent=4)
                print("File splits.json aggiornato con successo per le tile di FertyLab.")

if __name__ == "__main__":
    process_dataset_slicing()