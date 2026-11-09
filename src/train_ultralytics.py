import argparse
import csv
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from ultralytics import YOLO, RTDETR
from src.utils.config_loader import load_config


FITNESS_W = (0.1, 0.9)  # fitness Ultralytics = 0.1 * mAP50 + 0.9 * mAP50-95


def _write_fitness_column(csv_path: Path, epoch: int, fitness: float):
    if not csv_path.exists():
        return
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r]
    header, data = rows[0], rows[1:]
    if "fitness" not in header:
        header.append("fitness")
    fi, ei = header.index("fitness"), header.index("epoch")
    i50, i5095 = header.index("metrics/mAP50(B)"), header.index("metrics/mAP50-95(B)")
    for r in data:
        if len(r) <= fi:                       # riga senza fitness
            if int(float(r[ei])) == epoch:     # epoca corrente: valore registrato da Ultralytics durante il training
                r.append(f"{fitness:.5f}")
            else:                              # (solo in caso di resume di un run vecchio) ricostruita dalle mAP registrate
                r.append(f"{FITNESS_W[0] * float(r[i50]) + FITNESS_W[1] * float(r[i5095]):.5f}")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)


def make_fitness_callback():

    state = {"best": float("-inf"), "best_epoch": 0}

    def on_train_start(trainer):
        original_save_metrics = trainer.save_metrics

        def save_metrics_with_fitness(metrics):
            original_save_metrics(metrics)
            
            mAP50 = metrics.get("metrics/mAP50(B)", 0.0)
            mAP50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
            fit = float(0.1 * mAP50 + 0.9 * mAP50_95)
            
            
            trainer.fitness = fit
            if hasattr(trainer, "best_fitness"):
                if fit > trainer.best_fitness:
                    trainer.best_fitness = fit

            epoch = int(trainer.epoch) + 1
            _write_fitness_column(Path(trainer.csv), epoch, fit)
            if fit > state["best"]:
                state["best"], state["best_epoch"] = fit, epoch
            print(f"[FITNESS] epoca {epoch}/{trainer.epochs}: {fit:.4f} | migliore {state['best']:.4f} (epoca {state['best_epoch']})")

        trainer.save_metrics = save_metrics_with_fitness

    return on_train_start


def main():
    parser = argparse.ArgumentParser(description="Training per YOLOv8 e RT-DETR (stessi iperparametri comuni degli altri modelli)")
    parser.add_argument("--model", type=str, default="yolov8", choices=["yolov8", "rtdetr"])
    parser.add_argument("--regime", type=str, default="regime_b", choices=["regime_a", "regime_b"])
    parser.add_argument("--resume", action="store_true", help="Riprende da runs/train/<modello>_<regime>/weights/last.pt")
    args = parser.parse_args()

    cfg = load_config("configs/train_config.yaml")
    ds, tr, opt, sch = cfg["dataset"], cfg["training"], cfg["optimizer"], cfg["scheduler"]
    epochs, img_size = tr["epochs"], ds["img_size"]
    patience = tr.get("patience_ultralytics", tr.get("patience", 10))
    batch_size = ds.get("batch_size_ultralytics", 4)
    workers = int(ds.get("num_workers_ultralytics", ds.get("num_workers", 0))) 
    lr0 = float(opt["lr"])
    lrf = float(sch["eta_min"]) / lr0          # lr finale = lr0 * lrf = eta_min (cosine annealing)
    seed = int(cfg.get("project", {}).get("seed", 42))

    run_name = f"{args.model}_{args.regime}"
    last = Path("runs/train") / run_name / "weights" / "last.pt"
    data_yaml = f"configs/{args.regime}_yolo.yaml"

    print(f"\n[AVVIO TRAINING] {args.model.upper()} | {args.regime.upper()} | batch {batch_size} | workers {workers} (imposto da Ultralytics su MPS) | "
          f"AdamW lr0={lr0} -> {lr0 * lrf:g} (cosine) | patience {patience} | device MPS\n")

    if args.resume:
        if not last.exists():
            raise FileNotFoundError(f"--resume ma {last} non esiste")
        model = (YOLO if args.model == "yolov8" else RTDETR)(str(last))
        model.add_callback("on_train_start", make_fitness_callback())
        model.train(resume=True)
        return

    model = YOLO("yolov8n.pt") if args.model == "yolov8" else RTDETR("rtdetr-l.pt")
    model.add_callback("on_train_start", make_fitness_callback()) 
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device="mps",
        amp=False,                 
        workers=workers,
        optimizer="AdamW",
        lr0=lr0,
        lrf=lrf,
        cos_lr=True,
        weight_decay=float(opt["weight_decay"]),
        warmup_epochs=3,
        patience=patience,
        seed=seed,
        project=str(Path("runs/train").resolve()),   #
        name=run_name,
        exist_ok=True,
        save=True,
        plots=True,
        val=True,
    )
    print(f"\n[COMPLETATO] {args.model.upper()} ({args.regime.upper()}). Pesi migliori: runs/train/{run_name}/weights/best.pt")


if __name__ == "__main__":
    main()