import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

def convert_bbox_to_yolo(img_w: int, img_h: int, box: tuple) -> tuple:
    dw = 1.0 / img_w
    dh = 1.0 / img_h

    xmin, ymin, xmax, ymax = box

    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin

    x_center *= dw
    y_center *= dh
    w *= dw
    h *= dh

    return (
        round(x_center, 6),
        round(y_center, 6),
        round(w, 6),
        round(h, 6),
    )

# Recupera la directory del progetto e definisce i percorsi delle cartelle di input e output
def process_svia_dataset():
    project_root = Path(__file__).resolve().parent.parent

    svia_dir = project_root / "data" / "SVIA_DataSet"
    subset_a_dir = svia_dir / "Subset-A"
    frames_dir = svia_dir / "Frames from original videos"

    output_dir = project_root / "data" / "processed_SVIA"
    output_imgs = output_dir / "images"
    output_lbls = output_dir / "labels"

    output_imgs.mkdir(parents=True, exist_ok=True)
    output_lbls.mkdir(parents=True, exist_ok=True)

    xml_files = list(subset_a_dir.glob("*.xml"))
    print(f"Trovati {len(xml_files)} file XML di annotazione in Subset-A.")

    count = 0
    skipped = 0

    for xml_path in xml_files:
        stem = xml_path.stem 

        # Ricostruzione della cartella di origine del frame 
        folder_name = "_".join(stem.split("_")[:2])
        img_path = frames_dir / folder_name / f"{stem}.png"

        if not img_path.exists():

            matches = list(frames_dir.rglob(f"{stem}.png"))
            if matches:
                img_path = matches[0]
            else:
                skipped += 1
                continue

        # Lettura file XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size_elem = root.find("size")
        if (
            size_elem is not None
            and int(size_elem.find("width").text) > 0
            and int(size_elem.find("height").text) > 0
        ):
            w = int(size_elem.find("width").text)
            h = int(size_elem.find("height").text)
        else:
            with Image.open(img_path) as img:
                w, h = img.size

        yolo_lines = []
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            xc, yc, bw, bh = convert_bbox_to_yolo(w, h, (xmin, ymin, xmax, ymax))
            
            # Classe 0: Spermatozoo
            yolo_lines.append(f"0 {xc} {yc} {bw} {bh}\n")

        lbl_out_path = output_lbls / f"{stem}.txt"
        with open(lbl_out_path, "w", encoding="utf-8") as f:
            f.writelines(yolo_lines)

        img_out_path = output_imgs / img_path.name
        shutil.copy2(img_path, img_out_path)

        count += 1
        if count % 1000 == 0:
            print(f"Elaborati {count}/{len(xml_files)} file...")

    print(f"\n Convertite {count} immagini e annotazioni. Saltate: {skipped}.")
    print(f"Dataset salvato e pronto all'uso in: {output_dir}")

if __name__ == "__main__":
    process_svia_dataset()