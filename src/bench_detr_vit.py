import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
import torch.optim as optim

from src.models.vit_wrapper import ARCH_KEYS, BACKBONES, DETRViTWrapper

# immagini di training per epoca (dai log: regime A = 777 batch da 16; regime B = tile dei 3 campioni di train)
N_TRAIN = {"regime_a": 12432, "regime_b": 5179}


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def fake_batch(bs, size, device, n_boxes):
    x = torch.rand(bs, 3, size, size, device=device)
    labels = []
    for _ in range(bs):
        c = torch.rand(n_boxes, 2) * 0.9 + 0.05
        wh = torch.rand(n_boxes, 2) * 0.03 + 0.01
        labels.append({"boxes": torch.cat([c, wh], 1).to(device), "class_labels": torch.zeros(n_boxes, dtype=torch.long, device=device)})
    return x, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES), choices=list(BACKBONES))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--pool", type=int, default=2)
    ap.add_argument("--freeze-blocks", type=int, default=4)
    ap.add_argument("--n-boxes", type=int, default=20, help="spermatozoi per immagine (stima, influenza il matching)")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--no-pretrained", action="store_true", help="non scaricare i pesi (test solo di architettura/velocità)")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] device={device} batch={args.batch_size} img={args.img_size} pool={args.pool} freeze={args.freeze_blocks}")
    results = []

    for name in args.backbones:
        print(f"\n===== backbone: {name} ({BACKBONES[name]}) =====")
        t0 = time.time()
        model = DETRViTWrapper(num_classes=1, backbone=name, img_size=args.img_size, pool=args.pool,
                               freeze_blocks=args.freeze_blocks, init_transformer="coco",
                               load_pretrained=not args.no_pretrained).to(device)
        total, trainable = model.count_params()
        print(f"[1/3] costruito in {time.time() - t0:.0f}s | parametri {total / 1e6:.1f}M (addestrabili {trainable / 1e6:.1f}M)")

        opt = optim.AdamW(model.param_groups(1e-4, 1e-5, 5e-4))
        model.train()
        x, labels = fake_batch(args.batch_size, args.img_size, device, args.n_boxes)

        times = []
        for i in range(args.iters + 1):  # la prima iterazione è di warmup
            sync(device)
            t = time.time()
            opt.zero_grad(set_to_none=True)
            out = model(x, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            sync(device)
            if i > 0:
                times.append(time.time() - t)
            if not torch.isfinite(loss):
                raise RuntimeError("Loss non finita!")
        s_it = sum(times) / len(times)

        bb_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.model.model.backbone.parameters() if p.requires_grad)
        tr_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.model.model.encoder.parameters())
        print(f"[2/3] ok: loss={loss.item():.3f} | logits {tuple(out.logits.shape)} boxes {tuple(out.pred_boxes.shape)} | "
              f"gradiente backbone={bb_grad} transformer={tr_grad}")
        mem = ""
        if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            mem = f" | memoria MPS ~{torch.mps.current_allocated_memory() / 1e9:.1f} GB"
        print(f"      {s_it:.2f} s/iterazione{mem}")

        model.eval()
        with torch.no_grad():
            ref = model(x[:2]).logits.cpu()
        ckpt = {"model_state_dict": model.state_dict(), "arch": model.arch}
        torch.save(ckpt, "/tmp/_bench_detr_vit.pt")
        ck = torch.load("/tmp/_bench_detr_vit.pt", map_location=device)
        m2 = DETRViTWrapper(num_classes=1, load_pretrained=False, **{k: ck["arch"][k] for k in ARCH_KEYS}).to(device)
        m2.load_state_dict(ck["model_state_dict"], strict=True)
        m2.eval()
        with torch.no_grad():
            diff = (m2(x[:2]).logits.cpu() - ref).abs().max().item()
        print(f"[3/3] ricarica checkpoint: differenza massima sui logit = {diff:.2e} {'(OK)' if diff < 1e-4 else '(PROBLEMA!)'}")

        est = {r: n / args.batch_size * s_it / 3600 for r, n in N_TRAIN.items()}
        results.append((name, s_it, est))
        del model, m2, opt, out, loss
        if device.type == "mps":
            torch.mps.empty_cache()

    print("\n================ RIEPILOGO (solo training; la validation aggiunge ~10%) ================")
    print(f"{'backbone':10s} {'s/it':>7s} {'regime A /epoca':>17s} {'50 epoche':>11s} {'regime B /epoca':>17s}")
    for name, s_it, est in results:
        print(f"{name:10s} {s_it:7.2f} {est['regime_a']:14.1f} h {est['regime_a'] * 50 / 24:8.1f} gg {est['regime_b']:14.1f} h")
    print("\nBudget: ~1 h/epoca nel regime A (= DETR-R50 di prima: 2 giorni per 50 epoche). Scegli il backbone più grande che ci sta.")
    print("Nota: stima con immagini casuali e", args.n_boxes, "box per immagine; il tempo reale include il caricamento dati e può variare di ~10-20%.")


if __name__ == "__main__":
    main()