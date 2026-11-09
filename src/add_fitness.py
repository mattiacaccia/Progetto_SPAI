# Usato per aggiungere la colonna `fitness` ai file CSV della Detr-vit based dopo aver eseguito il training
import csv
import sys
from pathlib import Path


def add_fitness(path: Path) -> Path:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        rows = [[c.strip() for c in r] for r in reader if r]
    if "fitness" in header:
        raise ValueError(f"{path}: contiene già la colonna fitness")
    i50, i5095 = header.index("metrics/mAP50(B)"), header.index("metrics/mAP50-95(B)")
    out = path.with_name("results_with_fitness.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header + ["fitness"])
        for r in rows:
            fit = 0.1 * float(r[i50]) + 0.9 * float(r[i5095])
            w.writerow(r + [f"{fit:.5f}"])
    best = max(range(len(rows)), key=lambda k: 0.1 * float(rows[k][i50]) + 0.9 * float(rows[k][i5095]))
    print(f"{path} -> {out} | epoche {len(rows)} | fitness massima all'epoca {rows[best][header.index('epoch')]}")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        add_fitness(Path(p))