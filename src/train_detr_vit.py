
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # backend non interattivo (evita blocchi su macOS)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.utils.config_loader import load_config
from src.datasets.data_builder import get_train_val_dataloaders
from src.models.vit_wrapper import ARCH_KEYS, BACKBONES, DETRViTWrapper
from src.utils.det_metrics import compute_detection_metrics

CSV_HEADERS = [
    "epoch", "time",
    "train/giou_loss", "train/cls_loss", "train/l1_loss",
    "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    "val/giou_loss", "val/cls_loss", "val/l1_loss",
    "lr/pg0", "lr/pg1", "lr/pg2",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_targets(targets, device):
    out = []
    for t in targets:
        boxes = t.get("boxes", torch.zeros((0, 4)))
        labels = t.get("labels", torch.zeros((0,), dtype=torch.long))
        out.append({"boxes": boxes.to(device).float(), "class_labels": labels.to(device).long()})
    return out


def save_batch_images(dataloader, save_dir: Path):
    print("[ARTEFATTI] Salvataggio anteprime batch con bounding box...")
    for b_idx, (images, targets) in enumerate(dataloader):
        if b_idx >= 3:
            break
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()
        for i in range(min(4, len(images))):
            img_np = images[i].permute(1, 2, 0).cpu().numpy()
            ax = axes[i]
            ax.imshow(img_np)
            ax.axis('off')
            h_img, w_img = img_np.shape[:2]
            for box in targets[i].get("boxes", []):
                cx, cy, w, h = box.tolist()
                ax.add_patch(patches.Rectangle(((cx - w / 2) * w_img, (cy - h / 2) * h_img), w * w_img, h * h_img,
                                               linewidth=1.5, edgecolor='blue', facecolor='none'))
        plt.tight_layout()
        fig.savefig(save_dir / f"train_batch{b_idx}.jpg", dpi=150)
        plt.close(fig)


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    giou_tot = cls_tot = l1_tot = 0.0
    n_batches = 0
    all_preds, all_gts = [], []

    for images, targets in dataloader:
        images_tensor = torch.stack(images).to(device)
        outputs = model(images_tensor, labels=format_targets(targets, device))
        ld = outputs.loss_dict
        giou_tot += float(ld.get("loss_giou", 0.0))
        cls_tot += float(ld.get("loss_ce", 0.0))
        l1_tot += float(ld.get("loss_bbox", 0.0))
        n_batches += 1

        probs = outputs.logits.softmax(-1)[..., :-1]          # [B, Q, num_classi] (ultima = no-object)
        scores = probs.max(-1).values.float().cpu().numpy()    # [B, Q]
        cx, cy, w, h = outputs.pred_boxes.float().cpu().unbind(-1)
        pred_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1).clamp(0, 1).numpy()

        for b_i, t in enumerate(targets):
            gt = t.get("boxes", torch.zeros((0, 4))).float().cpu().numpy()
            if len(gt) > 0:
                gt_xyxy = np.stack([gt[:, 0] - gt[:, 2] / 2, gt[:, 1] - gt[:, 3] / 2,
                                    gt[:, 0] + gt[:, 2] / 2, gt[:, 1] + gt[:, 3] / 2], axis=1)
            else:
                gt_xyxy = np.zeros((0, 4))
            all_preds.append((pred_xyxy[b_i], scores[b_i]))
            all_gts.append(gt_xyxy)

    if n_batches == 0:
        raise RuntimeError("Validation vuota: controlla lo split in data_builder.py")

    avg_giou, avg_cls, avg_l1 = giou_tot / n_batches, cls_tot / n_batches, l1_tot / n_batches
    m = compute_detection_metrics(all_preds, all_gts, fixed_conf=0.5)
    precision, recall, _ = m["best_f1"]
    model.train()
    return avg_giou + avg_cls + avg_l1, avg_giou, avg_cls, avg_l1, precision, recall, m["mAP50"], m["mAP50-95"]


def generate_all_plots_and_metrics(save_dir: Path, csv_path: Path):
    if not csv_path.exists():
        return
    ep, tl, vl, pr, rc, m50, m5095 = [], [], [], [], [], [], []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ep.append(int(row["epoch"]))
            tl.append(float(row["train/giou_loss"]) + float(row["train/cls_loss"]) + float(row["train/l1_loss"]))
            vl.append(float(row["val/giou_loss"]) + float(row["val/cls_loss"]) + float(row["val/l1_loss"]))
            pr.append(float(row["metrics/precision(B)"]))
            rc.append(float(row["metrics/recall(B)"]))
            m50.append(float(row["metrics/mAP50(B)"]))
            m5095.append(float(row["metrics/mAP50-95(B)"]))
    if not ep:
        return
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("DETR ViT-based: training e validation (metriche reali su campione di validation isolato)", fontsize=14)
    for ax, (y, title, color) in zip(axs.flatten(), [
        (tl, "Train loss (giou+ce+l1)", "blue"), (vl, "Validation loss", "orange"),
        (pr, "Precision (soglia best-F1)", "green"), (rc, "Recall (soglia best-F1)", "red"),
        (m50, "mAP@0.5", "purple"), (m5095, "mAP@0.5:0.95", "magenta")]):
        ax.plot(ep, y, color=color)
        ax.set_title(title)
        ax.set_xlabel("epoca")
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "results.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Training DETR ViT-based (split per campione, validation reale)")
    parser.add_argument("--regime", type=str, default="regime_b", choices=["regime_a", "regime_b"])
    parser.add_argument("--backbone", type=str, default=None, choices=list(BACKBONES), help="Backbone ViT (default: da train_config.yaml)")
    parser.add_argument("--pool", type=int, default=None, help="Average pool 2x2 sulla griglia dei token (2 = 400 token per il transformer DETR)")
    parser.add_argument("--freeze-blocks", type=int, default=None, help="Blocchi ViT iniziali congelati")
    parser.add_argument("--init-transformer", type=str, default=None, choices=["coco", "random"])
    parser.add_argument("--lr", type=float, default=None, help="LR di transformer e teste")
    parser.add_argument("--backbone-lr", type=float, default=None, help="LR del backbone ViT")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Riprende da runs/train/detrvit_<regime>/weights/last.pt")
    args = parser.parse_args()

    cfg = load_config("train_config.yaml") if Path("train_config.yaml").exists() else load_config("configs/train_config.yaml")
    mcfg = (cfg.get("models", {}) or {}).get("detrvit", {}) or {}
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    set_seed(int(cfg.get("project", {}).get("seed", 42)))

    ds = cfg["dataset"]
    batch_size = args.batch_size or ds.get("batch_size_detrvit", ds["batch_size"])
    num_workers = int(ds.get("num_workers_detrvit", ds.get("num_workers", 0)))
    img_size = ds["img_size"]
    epochs = args.epochs or cfg["training"]["epochs"]
    patience = cfg["training"].get("patience_detrvit", cfg["training"].get("patience", 5))
    min_epochs = cfg["training"].get("min_epochs_detrvit", 10)
    opt = cfg["optimizer"]
    lr = args.lr or float(opt.get("detrvit_lr", 1e-4))
    backbone_lr = args.backbone_lr or float(opt.get("detrvit_backbone_lr", 1e-5))
    weight_decay = float(opt["weight_decay"])

    save_dir = Path("runs/train") / f"detrvit_{args.regime}"
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path, csv_path = weights_dir / "last.pt", weights_dir / "best.pt", save_dir / "results.csv"

    # ---- architettura: da checkpoint se si riprende, altrimenti da CLI/config ----
    ckpt = None
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"--resume ma {last_path} non esiste")
        ckpt = torch.load(last_path, map_location="cpu")
        arch = ckpt["arch"]
        print(f"[RESUME] Ripresa da {last_path} (epoca {ckpt['epoch']})")
    else:
        arch = dict(
            backbone=args.backbone or mcfg.get("backbone", "tiny"),
            img_size=img_size,
            pool=args.pool if args.pool is not None else int(mcfg.get("pool", 2)),
            freeze_blocks=args.freeze_blocks if args.freeze_blocks is not None else int(mcfg.get("freeze_blocks", 4)),
            num_queries=int(mcfg.get("num_queries", 100)),
            auxiliary_loss=bool(mcfg.get("auxiliary_loss", True)),
        )
    init_transformer = args.init_transformer or mcfg.get("init_transformer", "coco")

    print(f"\n[INIZIALIZZAZIONE] DETR ViT-based | {args.regime.upper()} | arch={arch}")
    train_loader, val_loader = get_train_val_dataloaders(
        regime=args.regime, batch_size=batch_size, img_size=img_size, num_workers=num_workers)
    if not args.resume:
        save_batch_images(train_loader, save_dir)

    model = DETRViTWrapper(num_classes=1, init_transformer=init_transformer,
                           load_pretrained=not args.resume, **{k: arch[k] for k in ARCH_KEYS}).to(device)
    total_p, train_p = model.count_params()
    print(f"[MODELLO] parametri totali {total_p/1e6:.1f}M | addestrabili {train_p/1e6:.1f}M")

    optimizer = optim.AdamW(model.param_groups(lr, backbone_lr, weight_decay))
    warmup_epochs = 3
    scheduler = SequentialLR(
        optimizer,
        schedulers=[LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
                    CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=float(cfg["scheduler"]["eta_min"]))],
        milestones=[warmup_epochs])

    start_epoch, best_fitness, patience_counter = 0, -1.0, 0
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"])
        best_fitness = float(ckpt.get("best_fitness", -1.0))
        patience_counter = int(ckpt.get("patience_counter", 0))
        del ckpt
    else:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f_csv:
            csv.writer(f_csv).writerow(CSV_HEADERS)

    print(f"\n[AVVIO TRAINING] batch={batch_size} workers={num_workers} lr={lr} backbone_lr={backbone_lr} "
          f"epoche={epochs} patience={patience} (attiva dopo l'epoca {min_epochs}) | device={device}\n")

    model.train()
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        g_tot = c_tot = l_tot = 0.0
        pbar = tqdm(train_loader, desc=f"Epoca [{epoch + 1}/{epochs}]", leave=True)
        for images, targets in pbar:
            images_tensor = torch.stack(images).to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images_tensor, labels=format_targets(targets, device))
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            ld = outputs.loss_dict
            g_tot += float(ld.get("loss_giou", 0.0))
            c_tot += float(ld.get("loss_ce", 0.0))
            l_tot += float(ld.get("loss_bbox", 0.0))
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        epoch_time = time.time() - t0
        n = max(1, len(train_loader))

        print(f"\n[VALIDAZIONE] epoca {epoch + 1}: metriche reali sul campione di validation...")
        val_loss, v_giou, v_cls, v_l1, precision, recall, map50, map50_95 = evaluate_model(model, val_loader, device)
        fitness = 0.1 * map50 + 0.9 * map50_95

        lrs = scheduler.get_last_lr()
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f_csv:
            csv.writer(f_csv).writerow([
                epoch + 1, f"{epoch_time:.2f}", f"{g_tot / n:.5f}", f"{c_tot / n:.5f}", f"{l_tot / n:.5f}",
                f"{precision:.5f}", f"{recall:.5f}", f"{map50:.5f}", f"{map50_95:.5f}",
                f"{v_giou:.5f}", f"{v_cls:.5f}", f"{v_l1:.5f}",
                f"{lrs[0]:.7f}", f"{lrs[1]:.7f}", "0.0000000"])

        print(f"Epoca [{epoch + 1}/{epochs}] | {epoch_time / 60:.1f} min | Val loss {val_loss:.4f} | "
              f"mAP50 {map50:.4f} | mAP50-95 {map50_95:.4f} | P {precision:.4f} | R {recall:.4f} | fitness {fitness:.4f}")

        improved = fitness > best_fitness + 1e-6
        if improved:
            best_fitness, patience_counter = fitness, 0
        elif epoch + 1 >= min_epochs:
            patience_counter += 1
            print(f"-> EarlyStopping: nessun miglioramento della fitness ({patience_counter}/{patience})")

        checkpoint = {
            "epoch": epoch + 1,
            "arch": arch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_fitness": best_fitness,
            "patience_counter": patience_counter,
            "fitness": fitness,
            "loss": val_loss,
        }
        torch.save(checkpoint, last_path)
        if improved:
            torch.save(checkpoint, best_path)
        if hasattr(torch, "mps") and device.type == "mps":
            torch.mps.empty_cache()

        if patience_counter >= patience:
            print(f"\n[EARLY STOPPING] nessun miglioramento per {patience} epoche (epoca {epoch + 1}).")
            break

    generate_all_plots_and_metrics(save_dir, csv_path)
    print(f"\n[COMPLETATO] DETR ViT-based ({args.regime.upper()}). Miglior fitness di validation: {best_fitness:.4f}")
    print(f"Pesi migliori: {best_path}")


if __name__ == "__main__":
    main()