import argparse
from pathlib import Path
import sys
import time
import csv
import yaml

import matplotlib
matplotlib.use('Agg') # Backend non interattivo per salvataggio sicuro su file
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
from torchvision.ops import nms
from ultralytics import YOLO, RTDETR

from src.utils.config_loader import load_config
from src.datasets.data_builder import get_dataloaders
from src.utils.det_metrics import compute_detection_metrics

# Import dei wrapper per i modelli PyTorch custom
from src.models.vit_wrapper import DETRViTWrapper
from src.models.tod_cnn_wrapper import TODCNNWrapper

def _decode_detr(outputs, b_i):

    probs = outputs.logits[b_i].softmax(-1)[..., :-1]
    scores = probs.max(-1).values
    cx, cy, w, h = outputs.pred_boxes[b_i].unbind(-1)
    xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1).clamp(0, 1)
    return xyxy.float().cpu(), scores.float().cpu()


def _decode_todcnn(pred_map, conf_thr=0.001, max_cand=300, nms_iou=0.5, max_det=100):
    pred_map = pred_map.detach().float().cpu()
    conf = torch.sigmoid(pred_map[4]).reshape(-1)
    top_conf, idx = torch.topk(conf, min(max_cand, conf.numel()))
    keep = top_conf > conf_thr
    top_conf, idx = top_conf[keep], idx[keep]
    if idx.numel() == 0:
        return torch.zeros((0, 4)), torch.zeros((0,))
    W = pred_map.shape[2]
    ys, xs = idx // W, idx % W
    cx, cy, w, h = pred_map[0, ys, xs], pred_map[1, ys, xs], pred_map[2, ys, xs], pred_map[3, ys, xs]
    xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1).clamp(0, 1)
    k = nms(xyxy, top_conf, nms_iou)[:max_det]
    return xyxy[k], top_conf[k]

def save_test_artifacts(images, gt_boxes_list, pred_boxes_list, save_dir: Path, model_name: str):

    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for i in range(min(4, len(images))):
        img_tensor = images[i]
        if img_tensor.ndim == 3 and img_tensor.shape[0] in [1, 3]:
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = img_tensor.cpu().numpy()
            
        ax = axes[i]
        ax.imshow(img_np)
        ax.axis('off')
        w_img, h_img = img_np.shape[1], img_np.shape[0]
        
        # Ground Truth (Blu)
        if i < len(gt_boxes_list):
            for box in gt_boxes_list[i]:
                x1, y1, x2, y2 = box.tolist()
                ax.add_patch(patches.Rectangle((x1*w_img, y1*h_img), (x2-x1)*w_img, (y2-y1)*h_img, linewidth=1.5, edgecolor='blue', facecolor='none'))
                
        # Predizioni (Rosso)
        if i < len(pred_boxes_list) and pred_boxes_list[i].numel() > 0:
            for box in pred_boxes_list[i]:
                x1, y1, x2, y2 = box.tolist()
                ax.add_patch(patches.Rectangle((x1*w_img, y1*h_img), (x2-x1)*w_img, (y2-y1)*h_img, linewidth=1.5, edgecolor='red', facecolor='none'))

    plt.tight_layout()
    fig.savefig(save_dir / f"test_inference_{model_name}.jpg", dpi=300)
    plt.close(fig)

def generate_train_vs_test_plots(train_dir: Path, test_dir: Path, test_metrics: dict, model_name: str, regime: str):
    train_csv = train_dir / "results.csv"
    if not train_csv.exists():
        print(f"[WARN] Impossibile generare il grafico comparativo: {train_csv} non trovato.")
        return

    epochs, train_losses, val_losses = [], [], []
    val_precisions, val_recalls, val_maps = [], [], []

    with open(train_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = {k.strip(): v.strip() for k, v in row.items()}
            ep = int(r.get("epoch", 0))
            epochs.append(ep)

            if "train/obj_loss" in r: # TOD-CNN
                t_l = float(r.get("train/obj_loss", 0)) + 5.0 * float(r.get("train/box_loss", 0))
                v_l = float(r.get("val/obj_loss", 0)) + 5.0 * float(r.get("val/box_loss", 0))
            elif "train/giou_loss" in r: # DETR ViT-based
                t_l = float(r.get("train/giou_loss", 0)) + float(r.get("train/cls_loss", 0)) + float(r.get("train/l1_loss", 0))
                v_l = float(r.get("val/giou_loss", 0)) + float(r.get("val/cls_loss", 0)) + float(r.get("val/l1_loss", 0))
            else: # Ultralytics (YOLO / RT-DETR)
                t_l = float(r.get("train/box_loss", 0)) + float(r.get("train/cls_loss", 0)) + float(r.get("train/dfl_loss", 0))
                v_l = float(r.get("val/box_loss", 0)) + float(r.get("val/cls_loss", 0)) + float(r.get("val/dfl_loss", 0))

            train_losses.append(t_l)
            val_losses.append(v_l)

            val_precisions.append(float(r.get("metrics/precision(B)", 0)))
            val_recalls.append(float(r.get("metrics/recall(B)", 0)))
            val_maps.append(float(r.get("metrics/mAP50(B)", 0)))

    # 1. Grafico Comparativo Metriche
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_precisions, label="Val Precision", color="green", linestyle="--", alpha=0.7)
    plt.plot(epochs, val_recalls, label="Val Recall", color="orange", linestyle="--", alpha=0.7)
    plt.plot(epochs, val_maps, label="Val mAP@0.5", color="purple", linewidth=2)

    plt.axhline(y=test_metrics["precision"], color="green", linestyle="-", linewidth=1.5, label=f"Test Precision ({test_metrics['precision']:.3f})")
    plt.axhline(y=test_metrics["recall"], color="orange", linestyle="-", linewidth=1.5, label=f"Test Recall ({test_metrics['recall']:.3f})")
    plt.axhline(y=test_metrics["map50"], color="purple", linestyle="-", linewidth=2, label=f"Test mAP@0.5 ({test_metrics['map50']:.3f})")

    plt.xlabel("Epoche di Training")
    plt.ylabel("Valore Metrica")
    plt.title(f"Confronto Metriche: Validation vs Test Set ({model_name.upper()} - {regime.upper()})")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(test_dir / "train_vs_test_metrics.png", dpi=300)
    plt.close()

    # 2. Grafico Loss
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_losses, label="Train Loss", color="blue", linewidth=2)
    plt.plot(epochs, val_losses, label="Val Loss", color="red", linewidth=2)
    plt.xlabel("Epoche")
    plt.ylabel("Loss")
    plt.title(f"Andamento Loss Training/Validation ({model_name.upper()} - {regime.upper()})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(test_dir / "train_vs_test_loss.png", dpi=300)
    plt.close()

    print(f"[INFO] Grafici comparativi salvati con successo in: {test_dir}")

def test_model_unified(model, test_loader, device, model_type, weights_path, regime, yolo_data=None):
    inference_images = []
    gt_boxes_all = []
    pred_boxes_all = []

    # CASO 1: YOLOv8 o RT-DETR 
    if model_type in ["yolov8", "rtdetr"]:
        data_yaml = yolo_data or f"configs/{regime}_yolo.yaml"
        print(f"[INFO] Esecuzione valutazione ufficiale Ultralytics su test set usando {data_yaml}...")
        
        metrics = model.val(data=data_yaml, split="test", verbose=False)
        
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        map50 = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

        # Raccogliamo un batch di immagini e predizioni per la visualizzazione grafica (save_test_artifacts)
        for images, targets in test_loader:
            images_tensor = torch.stack(images).to(device)
            results = model(images_tensor, verbose=False)
            for b_i, r in enumerate(results):
                inference_images.append(images[b_i])
                
                # GT conversion
                gt_raw = targets[b_i].get("boxes", torch.zeros((0, 4), device=device)).to(device)
                if gt_raw.numel() > 0:
                    cx, cy, w, h = gt_raw[:, 0], gt_raw[:, 1], gt_raw[:, 2], gt_raw[:, 3]
                    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                    gt_norm = torch.stack([x1, y1, x2, y2], dim=1)
                else:
                    gt_norm = torch.zeros((0, 4), device=device)
                gt_boxes_all.append(gt_norm)

                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    pred_boxes_all.append(boxes.xyxyn.to(device))
                else:
                    pred_boxes_all.append(torch.zeros((0, 4), device=device))
            break # Ne basta un batch per visualizzare i 4 campioni grafici

    # CASO 2: Modelli Custom PyTorch (DETR ViT-based, TOD-CNN) 
    else:
        model.eval()
        preds_np, gts_np = [], []
        with torch.no_grad():
            for images, targets in test_loader:
                outputs = model(torch.stack(images).to(device))
                for b_i, t in enumerate(targets):
                    gt_raw = t.get("boxes", torch.zeros((0, 4))).float().cpu()
                    if gt_raw.numel() > 0:
                        cx, cy, w, h = gt_raw.unbind(-1)
                        gt = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
                    else:
                        gt = torch.zeros((0, 4))

                    if model_type == "detrvit":
                        boxes, scores = _decode_detr(outputs, b_i)
                    else:
                        boxes, scores = _decode_todcnn(outputs[b_i])

                    preds_np.append((boxes.numpy(), scores.numpy()))
                    gts_np.append(gt.numpy())

                    if len(inference_images) < 4:  # servono solo per i grafici
                        inference_images.append(images[b_i].cpu())
                        gt_boxes_all.append(gt)
                        pred_boxes_all.append(boxes[scores >= 0.5])

        res = compute_detection_metrics(preds_np, gts_np, fixed_conf=0.5)
        precision, recall, f1_score = res["best_f1"]   # soglia che massimizza F1 (come Ultralytics)
        map50, map50_95 = res["mAP50"], res["mAP50-95"]
        p5, r5, f5 = res["fixed"]
        n_gt = sum(len(g) for g in gts_np)
        print(f"[INFO] GT totali: {n_gt} su {len(gts_np)} immagini ({n_gt / max(len(gts_np), 1):.1f} per immagine)")
        print(f"[INFO] Soglia best-F1: {res['best_conf']:.3f}")
        print(f"[INFO] A conf=0.5 -> P={p5:.4f} R={r5:.4f} F1={f5:.4f}")

    return precision, recall, f1_score, map50, map50_95, inference_images, gt_boxes_all, pred_boxes_all

def main():
    parser = argparse.ArgumentParser(description="Test Script Unificato con Metriche Ufficiali e Grafici Comparativi")
    parser.add_argument("--model", type=str, required=True, choices=["yolov8", "rtdetr", "detrvit", "todcnn"], help="Modello da testare")
    parser.add_argument("--regime", type=str, required=True, choices=["regime_a", "regime_b"], help="Regime di addestramento")
    parser.add_argument("--yolo-data", type=str, default="configs/test_yolo.yaml", help="YAML dataset per model.val() di YOLO/RT-DETR (es. configs/test_yolo.yaml con split test isolato)")
    args = parser.parse_args()

    cfg = load_config("train_config.yaml") if Path("train_config.yaml").exists() else load_config("configs/train_config.yaml")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[INFO] Esecuzione test su device: {device}")

    batch_size = cfg["dataset"]["batch_size"]
    img_size = cfg["dataset"]["img_size"]

    test_loader = get_dataloaders(regime="test", batch_size=batch_size, img_size=img_size)

    train_dir = Path(f"runs/train/{args.model}_{args.regime}")
    weights_path = train_dir / "weights" / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Pesi non trovati in {weights_path}. Completa prima il training!")

    if args.model == "yolov8":
        model = YOLO(str(weights_path))
    elif args.model == "rtdetr":
        model = RTDETR(str(weights_path))
    elif args.model == "detrvit":
        checkpoint = torch.load(weights_path, map_location=device)
        arch = checkpoint.get("arch", {}) if isinstance(checkpoint, dict) else {}
        # load_pretrained=False: nessun download, i pesi vengono tutti dal checkpoint
        model = DETRViTWrapper(num_classes=1, load_pretrained=False, **arch).to(device)
        sd = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(sd, strict=True)
        print(f"[INFO] DETR ViT-based caricato (arch: {arch}, epoca {checkpoint.get('epoch', '?')})")
    elif args.model == "todcnn":
        model = TODCNNWrapper(num_classes=1).to(device)
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)

    save_dir = Path("runs/test") / f"{args.model}_{args.regime}"
    save_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    precision, recall, f1_score, map50, map50_95, inf_imgs, gt_b, pred_b = test_model_unified(model, test_loader, device, args.model, weights_path, args.regime, args.yolo_data)
    duration = time.time() - start_time

    save_test_artifacts(inf_imgs, gt_b, pred_b, save_dir, args.model)

    # Salvataggio CSV ufficiale
    csv_path = save_dir / "test_results.csv"
    csv_headers = [
        "model", "regime", "time_sec", 
        "metrics/precision(B)", "metrics/recall(B)", "metrics/f1_score(B)", 
        "metrics/mAP50(B)", "metrics/mAP50-95(B)"
    ]
    
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(csv_headers)
        writer.writerow([
            args.model, args.regime, f"{duration:.2f}", 
            f"{precision:.5f}", f"{recall:.5f}", f"{f1_score:.5f}", 
            f"{map50:.5f}", f"{map50_95:.5f}"
        ])

    # Generazione dei Grafici Comparativi tra Training e Test
    test_metrics_dict = {
        "precision": precision,
        "recall": recall,
        "map50": map50
    }
    generate_train_vs_test_plots(train_dir, save_dir, test_metrics_dict, args.model, args.regime)

    print(f"\n[TEST COMPLETATO] Risultati Ufficiali {args.model.upper()} ({args.regime.upper()}):")
    print(f" - Precision: {precision:.4f}")
    print(f" - Recall:    {recall:.4f}")
    print(f" - F1-Score:  {f1_score:.4f}")
    print(f" - mAP@0.5:   {map50:.4f}")
    print(f" - mAP@0.5:0.95: {map50_95:.4f}")
    print(f"Artefatti e grafici comparativi salvati in: {save_dir}")

if __name__ == "__main__":
    main()