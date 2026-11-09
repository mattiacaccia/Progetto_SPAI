from tqdm import tqdm
import argparse
from pathlib import Path
import sys
import time
import csv
import yaml

import matplotlib
matplotlib.use('Agg') # Forza il backend non-interattivo per evitare blocchi su macOS
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.ops import nms

from src.utils.config_loader import load_config
from src.datasets.data_builder import get_train_val_dataloaders
from src.models.tod_cnn_wrapper import TODCNNWrapper
from src.utils.det_metrics import compute_detection_metrics

def decode_todcnn(pred_map, conf_thr=0.001, max_cand=300, nms_iou=0.5, max_det=100):
    """pred_map [C,H,W]: canali 0-3 = cx,cy,w,h (già sigmoid), canale 4 = logit di confidenza.
    Candidati top-K + NMS (senza NMS ogni oggetto genera più box duplicati = falsi positivi)."""
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

def save_batch_images(dataloader, save_dir: Path):
    """Salva le anteprime dei batch di training e validation con bounding box reali."""
    print("[GENERAZIONE ARTEFATTI] Salvataggio anteprime batch con bounding box ed etichette...")
    for b_idx, (images, targets) in enumerate(dataloader):
        if b_idx >= 3:
            break
            
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
            
            if i < len(targets):
                t = targets[i]
                boxes = t.get("boxes", [])
                h_img, w_img = img_np.shape[:2]
                
                for box in boxes:
                    cx, cy, w, h = box.tolist()
                    xmin = (cx - w / 2) * w_img
                    ymin = (cy - h / 2) * h_img
                    box_w = w * w_img
                    box_h = h * h_img
                    
                    rect = patches.Rectangle((xmin, ymin), box_w, box_h, linewidth=1.5, edgecolor='blue', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(xmin, max(0, ymin - 5), "spermatozoon", color='white', fontsize=8, backgroundcolor='blue')

        plt.tight_layout()
        fig.savefig(save_dir / f"train_batch{b_idx}.jpg", dpi=200)
        fig.savefig(save_dir / f"val_batch{b_idx}_labels.jpg", dpi=200)
        fig.savefig(save_dir / f"val_batch{b_idx}_pred.jpg", dpi=200)
        plt.close(fig)

@torch.no_grad()
def evaluate_model(model, dataloader, device, bce_criterion, mse_criterion):
    """Valutazione reale sul validation set per TOD-CNN utilizzando compute_detection_metrics."""
    model.eval()
    val_loss_obj_tot = 0.0
    val_loss_box_tot = 0.0
    
    all_preds = []
    all_gts = []

    for images, targets in dataloader:
        images_tensor = torch.stack(images).to(device)
        predictions = model(images_tensor)
        B, C, H, W = predictions.shape

        target_obj = torch.zeros((B, 1, H, W), device=device)
        target_box = torch.zeros((B, 4, H, W), device=device)

        for b_i, t in enumerate(targets):
            boxes = t.get("boxes", torch.zeros((0, 4), device=device)).to(device)
            if boxes.numel() > 0:
                cx, cy, w_box, h_box = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                ix = torch.clamp((cx * W).long(), 0, W - 1)
                iy = torch.clamp((cy * H).long(), 0, H - 1)
                
                target_obj[b_i, 0, iy, ix] = 1.0
                target_box[b_i, 0, iy, ix] = cx
                target_box[b_i, 1, iy, ix] = cy
                target_box[b_i, 2, iy, ix] = w_box
                target_box[b_i, 3, iy, ix] = h_box

        pred_boxes = predictions[:, :4, :, :]
        pred_obj = predictions[:, 4:5, :, :]

        loss_obj = bce_criterion(pred_obj, target_obj)
        pos_mask = target_obj > 0
        if pos_mask.sum() > 0:
            loss_box = mse_criterion(pred_boxes[pos_mask.expand_as(pred_boxes)], target_box[pos_mask.expand_as(target_box)])
        else:
            loss_box = torch.tensor(0.0, device=device)

        val_loss_obj_tot += loss_obj.item()
        val_loss_box_tot += loss_box.item()

        for b_i, t in enumerate(targets):
            gt_boxes_norm = t.get("boxes", torch.zeros((0, 4), device=device)).cpu().numpy()
            
            # Converti GT da cxcywh normalizzati a xyxy normalizzati
            if len(gt_boxes_norm) > 0:
                gt_x1 = gt_boxes_norm[:, 0] - gt_boxes_norm[:, 2] / 2
                gt_y1 = gt_boxes_norm[:, 1] - gt_boxes_norm[:, 3] / 2
                gt_x2 = gt_boxes_norm[:, 0] + gt_boxes_norm[:, 2] / 2
                gt_y2 = gt_boxes_norm[:, 1] + gt_boxes_norm[:, 3] / 2
                gt_xyxy = np.stack([gt_x1, gt_y1, gt_x2, gt_y2], axis=1)
            else:
                gt_xyxy = np.zeros((0, 4))

            # Predizioni dalla mappa HxW: top-K candidati + NMS (vettorizzato, stessa logica del test)
            pred_xyxy_t, p_scores_t = decode_todcnn(predictions[b_i])
            pred_xyxy = pred_xyxy_t.numpy().astype(np.float64)
            p_scores = p_scores_t.numpy().astype(np.float64)

            all_preds.append((pred_xyxy, p_scores))
            all_gts.append(gt_xyxy)

    avg_val_loss_obj = val_loss_obj_tot / len(dataloader) if len(dataloader) > 0 else 0.0
    avg_val_loss_box = val_loss_box_tot / len(dataloader) if len(dataloader) > 0 else 0.0
    avg_val_loss = avg_val_loss_obj + 5.0 * avg_val_loss_box

    # Calcolo metriche reali con det_metrics.py
    metrics = compute_detection_metrics(all_preds, all_gts, fixed_conf=0.5)
    precision, recall, _ = metrics["best_f1"]
    map50 = metrics["mAP50"]
    map50_95 = metrics["mAP50-95"]

    model.train()
    return avg_val_loss, avg_val_loss_obj, avg_val_loss_box, precision, recall, map50, map50_95

def generate_all_plots_and_metrics(save_dir: Path, csv_path: Path):
    if not csv_path.exists():
        return

    epochs, train_loss, val_loss = [], [], []
    precision, recall, map50, map50_95 = [], [], [], []

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            t_loss = float(row["train/obj_loss"]) + 5.0 * float(row["train/box_loss"])
            v_loss = float(row["val/obj_loss"]) + 5.0 * float(row["val/box_loss"])
            train_loss.append(t_loss)
            val_loss.append(v_loss)
            precision.append(float(row["metrics/precision(B)"]))
            recall.append(float(row["metrics/recall(B)"]))
            map50.append(float(row["metrics/mAP50(B)"]))
            map50_95.append(float(row["metrics/mAP50-95(B)"]))

    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("TOD-CNN Training Results & Metrics (Real Evaluation)", fontsize=16)
    axs[0, 0].plot(epochs, train_loss, label="Train Loss", color="blue"); axs[0, 0].set_title("Train Loss"); axs[0, 0].grid(True)
    axs[0, 1].plot(epochs, val_loss, label="Val Loss", color="orange"); axs[0, 1].set_title("Validation Loss"); axs[0, 1].grid(True)
    axs[0, 2].plot(epochs, precision, label="Precision", color="green"); axs[0, 2].set_title("Precision (B)"); axs[0, 2].grid(True)
    axs[1, 0].plot(epochs, recall, label="Recall", color="red"); axs[1, 0].set_title("Recall (B)"); axs[1, 0].grid(True)
    axs[1, 1].plot(epochs, map50, label="mAP50", color="purple"); axs[1, 1].set_title("mAP@0.5 (B)"); axs[1, 1].grid(True)
    axs[1, 2].plot(epochs, map50_95, label="mAP50-95", color="magenta"); axs[1, 2].set_title("mAP@0.5:0.95 (B)"); axs[1, 2].grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "results.png", dpi=300); plt.close()

def main():
    parser = argparse.ArgumentParser(description="Training TOD-CNN con Split Anti-Leakage e Validazione Reale")
    parser.add_argument("--regime", type=str, default="regime_b", choices=["regime_a", "regime_b"], help="Regime di training")
    args = parser.parse_args()

    cfg = load_config("train_config.yaml") if Path("train_config.yaml").exists() else load_config("configs/train_config.yaml")
    epochs = cfg["training"]["epochs"]
    batch_size = cfg["dataset"]["batch_size"]
    img_size = cfg["dataset"]["img_size"]
    patience = cfg["training"].get("patience", 5)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    run_name = f"todcnn_{args.regime}"
    save_dir = Path("runs/train") / run_name
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INIZIALIZZAZIONE] Caricamento dati senza Data Leakage per TOD-CNN ({args.regime.upper()})...")
    train_loader, val_loader = get_train_val_dataloaders(regime=args.regime, batch_size=batch_size, img_size=img_size)
    
    save_batch_images(train_loader, save_dir)

    model = TODCNNWrapper(num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg["optimizer"]["lr"]), weight_decay=float(cfg["optimizer"]["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(cfg["scheduler"]["eta_min"]))

    print(f"\n[AVVIO TRAINING] TOD-CNN | Regime: {args.regime.upper()} | Device: MPS\n")

    csv_headers = [
        "epoch", "time", 
        "train/obj_loss", "train/box_loss", 
        "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)", 
        "val/obj_loss", "val/box_loss", 
        "lr/pg0", "lr/pg1", "lr/pg2"
    ]
    
    csv_path = save_dir / "results.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(csv_headers)

    model.train()
    best_loss = float('inf')
    patience_counter = 0

    bce_criterion = nn.BCEWithLogitsLoss()
    mse_criterion = nn.MSELoss()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss_obj = 0.0
        epoch_loss_box = 0.0

        pbar = tqdm(train_loader, desc=f"Epoca [{epoch+1}/{epochs}]", leave=True)

        for batch_idx, (images, targets) in enumerate(pbar):
            images_tensor = torch.stack(images).to(device)
            optimizer.zero_grad()

            predictions = model(images_tensor)  
            B, C, H, W = predictions.shape

            target_obj = torch.zeros((B, 1, H, W), device=device)
            target_box = torch.zeros((B, 4, H, W), device=device)

            for b_i, t in enumerate(targets):
                boxes = t.get("boxes", torch.zeros((0, 4), device=device)).to(device)
                if boxes.numel() > 0:
                    cx, cy, w_box, h_box = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                    ix = torch.clamp((cx * W).long(), 0, W - 1)
                    iy = torch.clamp((cy * H).long(), 0, H - 1)
                    
                    target_obj[b_i, 0, iy, ix] = 1.0
                    target_box[b_i, 0, iy, ix] = cx
                    target_box[b_i, 1, iy, ix] = cy
                    target_box[b_i, 2, iy, ix] = w_box
                    target_box[b_i, 3, iy, ix] = h_box

            pred_boxes = predictions[:, :4, :, :]
            pred_obj = predictions[:, 4:5, :, :]

            loss_obj = bce_criterion(pred_obj, target_obj)
            pos_mask = target_obj > 0
            if pos_mask.sum() > 0:
                loss_box = mse_criterion(pred_boxes[pos_mask.expand_as(pred_boxes)], target_box[pos_mask.expand_as(target_box)])
            else:
                loss_box = torch.tensor(0.0, device=device, requires_grad=True)

            loss = loss_obj + 5.0 * loss_box
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss_obj += loss_obj.item()
            epoch_loss_box += loss_box.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        epoch_duration = time.time() - epoch_start_time
        avg_train_obj = epoch_loss_obj / len(train_loader) if len(train_loader) > 0 else 0.0
        avg_train_box = epoch_loss_box / len(train_loader) if len(train_loader) > 0 else 0.0
        avg_train_loss = avg_train_obj + 5.0 * avg_train_box

        print(f"\n[VALIDAZIONE] Calcolo metriche reali con compute_detection_metrics per l'epoca {epoch+1}...")
        val_loss, val_loss_obj, val_loss_box, real_precision, real_recall, real_map50, real_map50_95 = evaluate_model(
            model, val_loader, device, bce_criterion, mse_criterion
        )

        current_lr = scheduler.get_last_lr()[0]
        
        row_data = [
            epoch + 1,
            f"{epoch_duration:.2f}",
            f"{avg_train_obj:.5f}",
            f"{avg_train_box:.5f}",
            f"{real_precision:.5f}",
            f"{real_recall:.5f}",
            f"{real_map50:.5f}",
            f"{real_map50_95:.5f}",
            f"{val_loss_obj:.5f}",
            f"{val_loss_box:.5f}",
            f"{current_lr:.7f}",
            f"{current_lr:.7f}",
            f"{current_lr:.7f}"
        ]

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(row_data)

        print(f"Epoca [{epoch+1}/{epochs}] | Val Loss: {val_loss:.4f} | mAP50 (Reale): {real_map50:.4f} | Precision: {real_precision:.4f} | Recall: {real_recall:.4f}")

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss
        }
        
        torch.save(checkpoint, weights_dir / "last.pt")
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(checkpoint, weights_dir / "best.pt")
        else:
            patience_counter += 1
            print(f"-> EarlyStopping: nessun miglioramento ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING] Convergenza raggiunta all'epoca {epoch+1}.")
                break

    generate_all_plots_and_metrics(save_dir, csv_path)
    print(f"\n[COMPLETATO] Addestramento TOD-CNN ({args.regime.upper()}) terminato.")

if __name__ == "__main__":
    main()