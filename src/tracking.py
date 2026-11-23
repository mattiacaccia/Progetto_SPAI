
import argparse
import csv
import itertools
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

TILE_SIZE = 640
TILE_OVERLAP = 128                       # come slice_dataset.py
DEFAULT_FPS = 30.0
DEFAULT_UM_PER_PX = 0.18548383342803432
VIDEO_RE = re.compile(r"^(?P<sample>.+)_(?P<vid>V\d+)$")
TRAIN_SAMPLES = ["SEM1", "SEM-0003", "SEM-0007"]
VAL_SAMPLES = ["SEM-0004"]
MOTILITY_CLASSES = ["progressivo rapido", "progressivo lento", "non progressivo", "immobile"]
CLASS_COLORS = {"progressivo rapido": "#138a36", "progressivo lento": "#8cc63f", "non progressivo": "#e8590c",
                "immobile": "#2f5fb3"}
# parametri provati sulla validation
GRID = {"track_buffer": [10, 20, 30], "new_track": [0.3, 0.4, 0.5], "track_high": [0.3, 0.4, 0.5]}


# Video e camera.json
def discover_videos(videos_dir: Path):

    out = {}
    for sdir in sorted(p for p in videos_dir.iterdir() if p.is_dir()):
        for f in sorted(sdir.iterdir()):
            if f.is_file() and f.suffix.lower() == ".avi":
                m = VIDEO_RE.match(f.stem)
                if not m:
                    print(f"[AVVISO] nome non riconosciuto, salto: {f.relative_to(videos_dir)}")
                    continue
                js = f.with_name(f.stem + ".camera.json")
                out[(m["sample"], m["vid"])] = (f, js if js.exists() else None)
    return out


def read_camera(js):
    if js is None:
        return {}
    with open(js, encoding="utf-8") as f:
        d = json.load(f)
    prof, rec = d.get("profile", {}), d.get("recording", {})
    return {"fps": prof.get("fps"),
            "um_per_px": prof.get("microns_per_pixel"),     
            "timestamps": rec.get("frame_timestamps_seconds"),
            "capture_fps": rec.get("capture_fps")}


def frame_times(n, cam, fps):
    ts = cam.get("timestamps")
    if ts and len(ts) >= n:
        t = np.asarray(ts[:n], dtype=np.float64)
        return t - t[0]
    return np.arange(n, dtype=np.float64) / fps


def read_video_frames(video_path: Path):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video {video_path}")
    k = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield k, frame
            k += 1
    finally:
        cap.release()


def first_frame_rgb(video_path: Path):
    for _, f in read_video_frames(video_path):
        return f[:, :, ::-1].copy()
    return None


# Detection a tile (stessa griglia del training) e unione nel frame intero
def tile_grid(W, H, tile=TILE_SIZE, overlap=TILE_OVERLAP):
    stride, pos = tile - overlap, []
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            p = (max(0, min(x + tile, W) - tile), max(0, min(y + tile, H) - tile))
            if p not in pos:
                pos.append(p)
    return pos


def load_detector(model_name, weights_path, device, det_conf):
    from ultralytics import YOLO, RTDETR
    model = YOLO(str(weights_path)) if model_name == "yolov8" else RTDETR(str(weights_path))

    def detect(images):
        res = model.predict(images, imgsz=TILE_SIZE, conf=det_conf, iou=0.6, max_det=300, device=device, verbose=False)
        return [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()) if r.boxes is not None and len(r.boxes)
                else (np.zeros((0, 4)), np.zeros(0)) for r in res]

    return detect


def merge_nms(boxes, scores, iou_thr=0.5, ios_thr=1.01):
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        inter = ((np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest])).clip(0) *
                 (np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest])).clip(0))
        iou = inter / (areas[i] + areas[rest] - inter + 1e-12)
        ios = inter / (np.minimum(areas[i], areas[rest]) + 1e-12)
        order = rest[(iou < iou_thr) & (ios < ios_thr)]
    return np.array(keep, dtype=int)


def merge_tiles(tile_dets, offsets, W, H, iou_thr=0.5, ios_thr=1.01, edge_margin=2.0):
    # Riporta i box nel frame intero, scarta quelli tagliati da un bordo interno della tile e rimuove i doppioni
    all_b, all_s = [], []
    for (b, s), (ox, oy) in zip(tile_dets, offsets):
        if not len(s):
            continue
        b, s = np.asarray(b, dtype=np.float64).copy(), np.asarray(s, dtype=np.float64)
        if edge_margin >= 0:
            cut = (((ox > 0) & (b[:, 0] <= edge_margin)) | ((oy > 0) & (b[:, 1] <= edge_margin)) |
                   ((ox + TILE_SIZE < W) & (b[:, 2] >= TILE_SIZE - edge_margin)) |
                   ((oy + TILE_SIZE < H) & (b[:, 3] >= TILE_SIZE - edge_margin)))
            b, s = b[~cut], s[~cut]
        b[:, [0, 2]] += ox
        b[:, [1, 3]] += oy
        all_b.append(b)
        all_s.append(s)
    if not all_b or sum(len(s) for s in all_s) == 0:
        return np.zeros((0, 4)), np.zeros(0)
    boxes, scores = np.concatenate(all_b), np.concatenate(all_s)
    keep = merge_nms(boxes, scores, iou_thr, ios_thr)
    return boxes[keep], scores[keep]


def detect_all_frames(video_path: Path, detect, args):
    frame_dets, W, H = [], 0, 0
    for k, frame in read_video_frames(video_path):
        H, W = frame.shape[:2]
        grid = tile_grid(W, H)
        dets = detect([frame[y:y + TILE_SIZE, x:x + TILE_SIZE] for x, y in grid])
        boxes, scores = merge_tiles(dets, grid, W, H, args.merge_iou, args.merge_ios, args.edge_margin)
        frame_dets.append((k, boxes, scores))
    return frame_dets, W, H


def save_cache(path, frame_dets, W, H):
    np.savez_compressed(path, frames=np.array([k for k, _, _ in frame_dets]), W=W, H=H,
                        **{f"b{k}": b for k, b, _ in frame_dets}, **{f"s{k}": s for k, _, s in frame_dets})


def load_cache(path):
    z = np.load(path)
    return [(int(k), z[f"b{k}"], z[f"s{k}"]) for k in z["frames"]], int(z["W"]), int(z["H"])



# ByteTrack (Ultralytics) con associazione per distanza dei centri e tempo reale
class Dets:

    def __init__(self, xyxy, conf):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.zeros(len(self.conf), dtype=np.float32)

    @property
    def xywh(self):
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, mask):
        return Dets(self.xyxy[mask], self.conf[mask])


def make_tracker(p, px_per_frame_max):
    # p: dict con track_high, track_low, new_track, track_buffer, min_radius, max_gap_frames
    from ultralytics.trackers.byte_tracker import BYTETracker, STrack
    from ultralytics.trackers.basetrack import TrackState
    from ultralytics.trackers.utils import matching
    from ultralytics.utils import IterableSimpleNamespace

    cfg = IterableSimpleNamespace(tracker_type="bytetrack", track_high_thresh=p["track_high"],
                                  track_low_thresh=p["track_low"], new_track_thresh=p["new_track"],
                                  track_buffer=p["track_buffer"], match_thresh=1.0, fuse_score=False)

    class ObsSTrack(STrack):
        # ricorda posizione e istante (in frame nominali) dell'ultima detection associata
        current_frame = 0.0

        def _obs(self, src):
            self.last_obs = np.asarray(src.xywh[:2], dtype=np.float32).copy()
            self.last_obs_frame = ObsSTrack.current_frame

        def activate(self, kalman_filter, frame_id):
            super().activate(kalman_filter, frame_id)
            self._obs(self)

        def re_activate(self, new_track, frame_id, new_id=False):
            super().re_activate(new_track, frame_id, new_id)
            self._obs(new_track)

        def update(self, new_track, frame_id):
            super().update(new_track, frame_id)
            self._obs(new_track)

    class CenterDistanceBYTETracker(BYTETracker):
        track_class = ObsSTrack
        gap = 1

        def set_frame(self, t_frames, gap):
            ObsSTrack.current_frame = t_frames
            self.gap = gap

        def _center_cost(self, tracks, dets):
            # costo = distanza / R, con R = max(raggio minimo, v_max * tempo dall'ultima detection); > 1 = non associabile
            if len(tracks) == 0 or len(dets) == 0:
                return np.zeros((len(tracks), len(dets)), dtype=np.float32)
            dc = np.array([d.xywh[:2] for d in dets], dtype=np.float32)
            pc = np.array([t.xywh[:2] for t in tracks], dtype=np.float32)
            last = np.array([t.last_obs for t in tracks], dtype=np.float32)
            dt = np.array([min(max(1.0, ObsSTrack.current_frame - t.last_obs_frame), p["max_gap_frames"]) for t in tracks],
                          dtype=np.float32)
            R = np.maximum(p["min_radius"], px_per_frame_max * dt)[:, None]
            d_pred = np.linalg.norm(pc[:, None, :] - dc[None, :, :], axis=2)
            d_last = np.linalg.norm(last[:, None, :] - dc[None, :, :], axis=2)
            d = np.where(d_last <= R, np.minimum(d_pred, d_last), np.inf)
            return np.minimum(d / R, 1.01).astype(np.float32)

        def get_dists(self, tracks, detections):
            return self._center_cost(tracks, detections)

        def multi_predict(self, tracks):
            for _ in range(self.gap):          # un passo di Kalman per ogni frame trascorso, anche quelli persi
                STrack.multi_predict(tracks)

        def _second_association(self, strack_pool, u_track, detections_second, activated, refind, lost):
            r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
            if r_tracked and detections_second:
                dists = self._center_cost(r_tracked, detections_second)
                matches, u_track, _ = matching.linear_assignment(dists, thresh=1.0)
                self._apply_matches(matches, r_tracked, detections_second, activated, refind)
            else:
                u_track = list(range(len(r_tracked)))
            for it in u_track:
                r_tracked[it].mark_lost()
                lost.append(r_tracked[it])

    return CenterDistanceBYTETracker(cfg)


def run_tracker(frame_dets, times, p, um_per_px, fps):
    # Il tracker lavora in frame nominali del tipo tempo x fps così un frame perso dalla camera conta come due passi e il raggio cresce di conseguenza.
    tracker = make_tracker(p, p["max_speed"] / um_per_px / fps)
    rows, prev = [], None
    for k, boxes, scores in frame_dets:
        tn = float(times[k]) * fps
        tracker.set_frame(tn, 1 if prev is None else max(1, int(round(tn - prev))))
        prev = tn
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        out = tracker.update(Dets(boxes, scores))
        for r in (out if len(out) else []):
            # posizione = box della detection associata (non il box del filtro di Kalman)
            x1, y1, x2, y2 = boxes[int(r[7])] if len(r) > 7 and 0 <= int(r[7]) < len(boxes) else r[:4]
            rows.append((k, int(r[4]), float(x1), float(y1), float(x2), float(y2), float(r[5])))
    return rows


def tracks_of(rows):
    tr = defaultdict(list)
    for k, tid, x1, y1, x2, y2, _ in rows:
        tr[tid].append((k, (x1 + x2) / 2.0, (y1 + y2) / 2.0))
    return {tid: sorted(pts) for tid, pts in tr.items()}


# CASA
def motility_class(vcl, vap, args):
    # OMS 2021 (stimata dalla testa)
    if vcl < args.static_vcl:
        return "immobile"
    if vap < args.slow_vap:
        return "non progressivo"
    if vap < args.rapid_vap:
        return "progressivo lento"
    return "progressivo rapido"


def smooth_path(xy, w=7):
    # media mobile centrata su w punti 
    half, n = (w - 1) // 2, len(xy)
    out, full = np.empty_like(xy), np.zeros(n, dtype=bool)
    for i in range(n):
        h = min(half, i, n - 1 - i)
        out[i] = xy[i - h:i + h + 1].mean(axis=0)
        full[i] = h == half
    return out, full


def beat_cross_frequency(xy, sp, full, t):
    idx = np.where(full)[0]
    if len(idx) < 3:
        return None
    tang = np.gradient(sp, axis=0)
    dev = xy - sp
    side = np.sign(tang[:, 0] * dev[:, 1] - tang[:, 1] * dev[:, 0])[idx]
    side = side[side != 0]
    T = t[idx[-1]] - t[idx[0]]
    if len(side) < 2 or T <= 0:
        return None
    return float((side[1:] != side[:-1]).sum() / T)


def compute_casa(rows, times, um_per_px, args):
    out, eps = [], 1e-9
    for tid, pts in tracks_of(rows).items():
        if len(pts) < args.min_track_len:
            continue
        t = np.array([times[k] for k, _, _ in pts], dtype=np.float64)
        xy = np.array([(x, y) for _, x, y in pts], dtype=np.float64) * um_per_px
        T = t[-1] - t[0]
        if T <= 0:
            continue
        vcl = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() / T
        vsl = np.linalg.norm(xy[-1] - xy[0]) / T
        sp, full = smooth_path(xy, args.vap_window)
        vap = min(np.linalg.norm(np.diff(sp, axis=0), axis=1).sum() / T, vcl)       # VSL <= VAP <= VCL
        dev = np.linalg.norm(xy - sp, axis=1)
        bcf = beat_cross_frequency(xy, sp, full, t)
        out.append({
            "track_id": tid, "n_frame": len(pts), "frame_inizio": pts[0][0], "frame_fine": pts[-1][0],
            "durata_s": round(float(T), 3), "VCL": round(float(vcl), 2), "VSL": round(float(vsl), 2),
            "VAP": round(float(vap), 2), "LIN_pct": round(float(vsl / (vcl + eps) * 100), 1),
            "STR_pct": round(float(vsl / (vap + eps) * 100), 1), "WOB_pct": round(float(vap / (vcl + eps) * 100), 1),
            "ALH": round(float(dev[full].mean() if full.any() else dev.mean()), 2),
            "BCF": round(bcf, 2) if bcf is not None else None,
            "classe": motility_class(vcl, vap, args),
        })
    return out


def track_stats(rows, frame_dets, min_len, ref_score):
    tr = tracks_of(rows)
    L = np.array([len(p) for p in tr.values()]) if tr else np.zeros(0, dtype=int)
    long_ids = {tid for tid, p in tr.items() if len(p) >= min_len}
    n_frames = max(len(frame_dets), 1)
    n_ref = sum(int((np.asarray(s) >= ref_score).sum()) for _, _, s in frame_dets)
    # punti delle tracce lunghe che provengono da detection affidabili (stessa soglia del denominatore)
    pts_long = sum(1 for r in rows if r[1] in long_ids and r[6] >= ref_score)
    Ll = L[L >= min_len]
    cont = float((Ll ** 2).sum() / Ll.sum() / n_frames) if len(Ll) else 0.0
    cov = pts_long / max(n_ref, 1)
    return {"frame": len(frame_dets), "detection_per_frame": round(n_ref / n_frames, 2), "tracce": int(len(L)),
            "tracce_lunghe": len(long_ids), "lunghezza_mediana": float(np.median(L)) if len(L) else 0.0,
            "copertura": round(cov, 4), "continuita": round(cont, 4), "punteggio": round(cov * cont, 4)}


# Grafici
def plot_trajectories(rows, casa, background, title, out_path, args):
    # linea spessa = percorso medio (VAP); tratteggio = percorso puntuale (VCL); pallino = inizio; freccia = fine
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cls = {c["track_id"]: c["classe"] for c in casa}
    H, W = background.shape[:2] if background is not None else (720, 1280)
    fig, ax = plt.subplots(figsize=(15.5, 13 * H / W + 0.8))
    if background is not None:
        ax.imshow(background, cmap="gray", alpha=0.5)
    counts = defaultdict(int)
    for tid, pts in tracks_of(rows).items():
        if tid not in cls:
            continue
        a = np.array([(x, y) for _, x, y in pts])
        sm, _ = smooth_path(a, args.vap_window)
        c = CLASS_COLORS[cls[tid]]
        counts[cls[tid]] += 1
        ax.plot(a[:, 0], a[:, 1], "--", color=c, linewidth=0.8, alpha=0.7)
        ax.plot(sm[:, 0], sm[:, 1], "-", color="white", linewidth=4.2, alpha=0.85, solid_capstyle="round")
        ax.plot(sm[:, 0], sm[:, 1], "-", color=c, linewidth=2.4, solid_capstyle="round")
        ax.plot(a[0, 0], a[0, 1], "o", color=c, markersize=6, markeredgecolor="white", markeredgewidth=1)
        ax.text(a[0, 0] + 6, a[0, 1] - 6, str(tid), fontsize=7.5, color=c, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
        if np.linalg.norm(sm[-1] - sm[-2]) > 1:
            ax.annotate("", xy=sm[-1], xytext=sm[-2], arrowprops=dict(arrowstyle="-|>", color=c, lw=2, mutation_scale=14))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_title(title, fontsize=12)
    handles = [Line2D([0], [0], color=CLASS_COLORS[k], lw=2.5, label=f"{k} ({counts.get(k, 0)})") for k in CLASS_COLORS]
    handles += [Line2D([0], [0], color="#555555", lw=2.4, label="percorso medio (VAP)"),
                Line2D([0], [0], color="#555555", lw=0.8, ls="--", label="percorso puntuale (VCL)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=False,
              title=f"tracce ≥ {args.min_track_len} frame")
    fig.text(0.01, 0.005, f"Sfondo: primo frame. Classi OMS 2021: immobile VCL < {args.static_vcl:g}; non progressivo "
             f"VAP < {args.slow_vap:g}; progressivo lento VAP < {args.rapid_vap:g}; progressivo rapido VAP >= "
             f"{args.rapid_vap:g} µm/s.", fontsize=8, color="#555555")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def make_gif(video_path: Path, rows, out_path, step=3, scale=0.4, tail=15, fps_gif=10):
    import cv2
    from PIL import Image
    by_f = defaultdict(list)
    for k, tid, x1, y1, x2, y2, _ in rows:
        by_f[k].append((tid, x1, y1, x2, y2))
    hist, rng, imgs = defaultdict(list), np.random.default_rng(0), []
    col = defaultdict(lambda: tuple(int(v) for v in rng.integers(40, 230, 3)))
    for k, frame in read_video_frames(video_path):
        for tid, x1, y1, x2, y2 in by_f.get(k, []):
            hist[tid].append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
        if k % step:
            continue
        im = np.ascontiguousarray(frame[:, :, ::-1])
        for tid, x1, y1, x2, y2 in by_f.get(k, []):
            c = col[tid]
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
            cv2.putText(im, str(tid), (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
            pts = hist[tid][-tail:]
            for a, b in zip(pts, pts[1:]):
                cv2.line(im, a, b, c, 2)
        cv2.putText(im, f"frame {k}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(im, f"frame {k}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
        imgs.append(Image.fromarray(cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)))
    if imgs:
        imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps_gif), loop=0)


def export_pseudo_labels(rows, casa, tag, W, H, out_dir: Path):
    keep = {c["track_id"] for c in casa}
    mot_dir, yolo_dir = out_dir / "pseudo_labels" / "mot", out_dir / "pseudo_labels" / "yolo" / tag
    mot_dir.mkdir(parents=True, exist_ok=True)
    yolo_dir.mkdir(parents=True, exist_ok=True)
    by_f, n = defaultdict(list), 0
    with open(mot_dir / f"{tag}.txt", "w") as f:
        for k, tid, x1, y1, x2, y2, sc in sorted(rows):
            if tid in keep:
                f.write(f"{k + 1},{tid},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{sc:.3f},-1,-1,-1\n")
                by_f[k].append((x1, y1, x2, y2))
                n += 1
    for k, bxs in by_f.items():
        with open(yolo_dir / f"{tag}_frame_{k:04d}.txt", "w") as f:
            for x1, y1, x2, y2 in bxs:
                f.write(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
    return n, len(by_f)


def write_dicts(path, dict_rows):
    if not dict_rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict_rows[0].keys()))
        w.writeheader()
        w.writerows(dict_rows)


def summarize(casa_rows, keys_group):
    out = []
    groups = defaultdict(list)
    for r in casa_rows:
        groups[tuple(r[k] for k in keys_group)].append(r)
    order = {"train": 0, "val": 1, "test": 2}
    for g, rs in sorted(groups.items(), key=lambda it: (order.get(it[0][0], 9), it[0])):
        n = len(rs)
        row = dict(zip(keys_group, g))
        row.update({"video": len({(r["campione"], r["video"]) for r in rs}), "tracce": n})
        for c in MOTILITY_CLASSES:
            row[f"{c} %"] = round(100 * sum(r["classe"] == c for r in rs) / n, 1)
        row["progressivi %"] = round(row["progressivo rapido %"] + row["progressivo lento %"], 1)
        for k in ("VCL", "VSL", "VAP", "LIN_pct", "STR_pct", "WOB_pct", "ALH", "BCF"):
            v = [r[k] for r in rs if r[k] is not None]
            row[f"{k} mediana"] = round(float(np.median(v)), 2) if v else None
        out.append(row)
    return out


# Main

def main():
    ap = argparse.ArgumentParser(description="ByteTrack + CASA sui video di data/spai_video")
    ap.add_argument("--model", required=True, choices=["yolov8", "rtdetr"])
    ap.add_argument("--regime", required=True, choices=["regime_a", "regime_b"])
    ap.add_argument("--stage", required=True, choices=["detect", "tune", "casa"])
    ap.add_argument("--weights", default=None, help="Default: runs/train/<model>_<regime>/weights/best.pt")
    ap.add_argument("--videos-dir", default="data/spai_video")
    ap.add_argument("--train-samples", nargs="+", default=TRAIN_SAMPLES)
    ap.add_argument("--val-samples", nargs="+", default=VAL_SAMPLES)
    ap.add_argument("--only", nargs="+", default=None, help="Solo questi campioni (es. --only SEM-0004)")
    ap.add_argument("--exclude", nargs="+", default=["SEM-0012", "SEM-0013"],
                    help="Campioni esclusi (default: SEM-0012 e SEM-0013, senza spermatozoi come nell'object detection). "
                         "Passa --exclude senza valori per non escludere nulla")
    ap.add_argument("--device", default=None, help="mps | cpu (default: mps se disponibile)")
    # detection
    ap.add_argument("--det-conf", type=float, default=0.1)
    ap.add_argument("--merge-iou", type=float, default=0.5)
    ap.add_argument("--merge-ios", type=float, default=1.01)
    ap.add_argument("--edge-margin", type=float, default=2.0)
    # tracker (i valori scelti in tune sovrascrivono questi)
    ap.add_argument("--max-speed", type=float, default=None, help="v_max (µm/s); default: stimata sul train in tune")
    ap.add_argument("--speed-percentile", type=float, default=95.0)
    ap.add_argument("--min-radius", type=float, default=10.0, help="Raggio minimo di associazione (px)")
    ap.add_argument("--max-gap-frames", type=int, default=30)
    ap.add_argument("--track-high", type=float, default=0.3)
    ap.add_argument("--track-low", type=float, default=0.1)
    ap.add_argument("--new-track", type=float, default=0.3)
    ap.add_argument("--track-buffer", type=int, default=20)
    # CASA
    ap.add_argument("--min-track-len", type=int, default=30, help="Frame minimi per il CASA (30 ≈ 1 s)")
    ap.add_argument("--vap-window", type=int, default=7)
    ap.add_argument("--static-vcl", type=float, default=5.0)
    ap.add_argument("--slow-vap", type=float, default=5.0)
    ap.add_argument("--rapid-vap", type=float, default=25.0)
    # grafici
    ap.add_argument("--plot", choices=["none", "sample", "all"], default="all",
                    help="Grafico delle traiettorie: nessuno, uno per campione, uno per video")
    ap.add_argument("--gif", action="store_true", help="Anche una GIF con box e ID (stessi video del grafico)")
    ap.add_argument("--casa-splits", nargs="+", default=["test"], choices=["train", "val", "test"],
                    help="Insiemi su cui fare tracking, pseudo-etichette e CASA (default: solo test)")
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir)
    if not videos_dir.exists():
        raise FileNotFoundError(f"Cartella dei video non trovata: {videos_dir.resolve()}")
    out_dir = Path("runs/tracking_video") / f"{args.model}_{args.regime}"
    for sub in ("det_cache", "tracks", "trajectories"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    videos = discover_videos(videos_dir)
    if args.exclude:
        videos = {k: v for k, v in videos.items() if k[0] not in args.exclude}
        print(f"[INFO] Campioni esclusi: {', '.join(args.exclude)}")
    if args.only:
        videos = {k: v for k, v in videos.items() if k[0] in args.only}
    cams = {k: read_camera(js) for k, (_, js) in videos.items()}
    split = {k: "train" if k[0] in args.train_samples else "val" if k[0] in args.val_samples else "test" for k in videos}
    for s in ("train", "val", "test"):
        ks = [k for k in videos if split[k] == s]
        print(f"[INFO] {s:5s}: {len(ks):3d} video | campioni: {', '.join(sorted({k[0] for k in ks})) or '-'}")
    no_json = [f"{k[0]}_{k[1]}" for k, (_, js) in videos.items() if js is None]
    if no_json:
        print(f"[AVVISO] senza .camera.json (uso {DEFAULT_FPS:g} fps e {DEFAULT_UM_PER_PX:.4f} µm/px): {', '.join(no_json)}")

    def cache_of(k):
        return out_dir / "det_cache" / f"{k[0]}_{k[1]}.npz"

    # detect
    if args.stage == "detect":
        if args.device:
            device = args.device
        else:
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        weights = Path(args.weights) if args.weights else Path(f"runs/train/{args.model}_{args.regime}/weights/best.pt")
        if not weights.exists():
            raise FileNotFoundError(f"Pesi non trovati: {weights}")
        detect = load_detector(args.model, weights, device, args.det_conf)
        keys = sorted(videos)
        for i, k in enumerate(keys):
            if cache_of(k).exists():
                print(f"[{i + 1}/{len(keys)}] {k[0]}_{k[1]}: già fatto")
                continue
            t0 = time.time()
            dets, W, H = detect_all_frames(videos[k][0], detect, args)
            save_cache(cache_of(k), dets, W, H)
            print(f"[{i + 1}/{len(keys)}] {k[0]}_{k[1]} [{split[k]}]: {len(dets)} frame {W}x{H}, "
                  f"{np.mean([len(s) for _, _, s in dets]):.1f} detection/frame | {time.time() - t0:.0f} s")
        print(f"\n[OK] Detection salvate in {out_dir / 'det_cache'}")
        return

    def load(k):
        if not cache_of(k).exists():
            return None
        dets, W, H = load_cache(cache_of(k))
        cam = cams[k]
        fps = cam.get("fps") or DEFAULT_FPS
        return dets, frame_times(len(dets), cam, fps), cam.get("um_per_px") or DEFAULT_UM_PER_PX, fps, (W, H)

    missing = [f"{k[0]}_{k[1]}" for k in videos if not cache_of(k).exists()]
    if missing:
        print(f"[AVVISO] {len(missing)} video senza detection (lancia prima --stage detect): {', '.join(missing[:8])}"
              + (" ..." if len(missing) > 8 else ""))

    base = {"max_speed": args.max_speed, "min_radius": args.min_radius, "max_gap_frames": args.max_gap_frames,
            "track_high": args.track_high, "track_low": args.track_low, "new_track": args.new_track,
            "track_buffer": args.track_buffer}

    # tune
    if args.stage == "tune":
        train = [k for k in sorted(videos) if split[k] == "train" and cache_of(k).exists()]
        val = [k for k in sorted(videos) if split[k] == "val" and cache_of(k).exists()]
        if not train or not val:
            raise RuntimeError("Servono le detection di train e validation: lancia prima --stage detect")

        # 1) v_max dal train: tracking con raggio largo, velocità istantanee delle tracce lunghe
        if args.max_speed is None:
            speeds = []
            p0 = {**base, "max_speed": 150.0}
            for k in train:
                dets, times, um, fps, _ = load(k)
                for pts in tracks_of(run_tracker(dets, times, p0, um, fps)).values():
                    if len(pts) >= args.min_track_len:
                        t = np.array([times[f] for f, _, _ in pts])
                        xy = np.array([(x, y) for _, x, y in pts]) * um
                        dt = np.diff(t)
                        ok = dt > 0
                        speeds += list(np.linalg.norm(np.diff(xy, axis=0), axis=1)[ok] / dt[ok])
            if len(speeds) < 50:
                raise RuntimeError("Troppo poche tracce nel train per stimare v_max: passa --max-speed a mano (es. 60)")
            v_max = round(float(np.percentile(speeds, args.speed_percentile)), 1)
            print(f"[INFO] Train ({len(train)} video): {len(speeds)} spostamenti tra frame consecutivi, mediana "
                  f"{np.median(speeds):.1f} µm/s, {args.speed_percentile:g}° percentile {v_max} µm/s -> v_max")
        else:
            v_max = args.max_speed
            print(f"[INFO] v_max impostata a mano: {v_max} µm/s")

        # 2) grid search sulla validation: punteggio = copertura x continuità (media sui video)
        val_data = {k: load(k) for k in val}
        results, combos = [], list(itertools.product(*GRID.values()))
        t0 = time.time()
        for vals in combos:
            p = {**base, "max_speed": v_max, **dict(zip(GRID, vals))}
            st = [track_stats(run_tracker(d, t, p, um, fps), d, args.min_track_len, args.track_high)
                  for d, t, um, fps, _ in val_data.values()]
            results.append({**dict(zip(GRID, vals)),
                            **{m: round(float(np.mean([s[m] for s in st])), 4)
                               for m in ("tracce_lunghe", "copertura", "continuita", "punteggio")}})
        results.sort(key=lambda r: -r["punteggio"])
        write_dicts(out_dir / "tuning_validation.csv", results)
        best = {**base, "max_speed": v_max, **{g: results[0][g] for g in GRID}}
        with open(out_dir / "tracker_params.json", "w", encoding="utf-8") as f:
            json.dump({"parametri": best, "validation": results[0], "train_samples": args.train_samples,
                       "val_samples": args.val_samples}, f, indent=2, ensure_ascii=False)
        print(f"[INFO] {len(combos)} combinazioni provate su {len(val)} video di validation in {time.time() - t0:.0f} s")
        print("\n" + "=" * 80)
        print(f" PARAMETRI SCELTI: v_max {v_max} µm/s | " + " | ".join(f"{g} {best[g]}" for g in GRID))
        print(f" validation: copertura {results[0]['copertura']:.3f} | continuità {results[0]['continuita']:.3f} | "
              f"punteggio {results[0]['punteggio']:.3f}")
        print("=" * 80)
        print(f"[OK] {out_dir / 'tracker_params.json'}")
        return

    # casa
    pfile = out_dir / "tracker_params.json"
    if pfile.exists():
        with open(pfile, encoding="utf-8") as f:
            p = json.load(f)["parametri"]
        print(f"[INFO] Parametri da {pfile}: {p}")
    else:
        p = {**base, "max_speed": args.max_speed or 60.0}
        print(f"[AVVISO] {pfile} non trovato (lancia prima --stage tune): uso i default {p}")

    casa_rows, stat_rows, plotted = [], [], set()
    keys = [k for k in sorted(videos, key=lambda k: ({"train": 0, "val": 1, "test": 2}[split[k]], k))
            if split[k] in args.casa_splits and cache_of(k).exists()]
    if not keys:
        raise RuntimeError(f"Nessun video di {'/'.join(args.casa_splits)} con le detection: lancia prima --stage detect")
    n_pseudo = [0, 0]
    for i, k in enumerate(keys):
        dets, times, um, fps, (W, H) = load(k)
        tag = f"{k[0]}_{k[1]}"
        rows = run_tracker(dets, times, p, um, fps)
        casa = compute_casa(rows, times, um, args)
        st = track_stats(rows, dets, args.min_track_len, p["track_high"])
        stat_rows.append({"insieme": split[k], "campione": k[0], "video": k[1], "durata_s": round(float(times[-1]), 2), **st})
        casa_rows += [{"insieme": split[k], "campione": k[0], "video": k[1], **c} for c in casa]
        with open(out_dir / "tracks" / f"{tag}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "tempo_s", "track_id", "x1", "y1", "x2", "y2", "score"])
            w.writerows([(r[0], round(float(times[r[0]]), 4), *r[1:]) for r in rows])
        nb, nf = export_pseudo_labels(rows, casa, tag, W, H, out_dir)
        n_pseudo[0] += nb
        n_pseudo[1] += nf
        n = defaultdict(int)
        for c in casa:
            n[c["classe"]] += 1
        print(f"[{i + 1}/{len(keys)}] {tag} [{split[k]}]: {st['tracce_lunghe']} tracce CASA | rapidi {n['progressivo rapido']}, "
              f"lenti {n['progressivo lento']}, non progressivi {n['non progressivo']}, immobili {n['immobile']}")
        if args.plot == "all" or (args.plot == "sample" and k[0] not in plotted):
            plotted.add(k[0])
            plot_trajectories(rows, casa, first_frame_rgb(videos[k][0]),
                              f"{tag} [{split[k]}] - {args.model.upper()} {args.regime} - ByteTrack - "
                              f"{len(dets)} frame in {times[-1]:.1f} s", out_dir / "trajectories" / f"{tag}.png", args)
            if args.gif:
                make_gif(videos[k][0], rows, out_dir / "trajectories" / f"{tag}.gif")

    write_dicts(out_dir / "casa_tracce.csv", casa_rows)
    write_dicts(out_dir / "statistiche_tracce.csv", stat_rows)
    per_sample = summarize(casa_rows, ("insieme", "campione"))
    per_split = summarize(casa_rows, ("insieme",))
    write_dicts(out_dir / "casa_per_campione.csv", per_sample)
    write_dicts(out_dir / "casa_per_insieme.csv", per_split)

    print("\n" + "=" * 100)
    print(f" CASA (tracce con >= {args.min_track_len} frame; velocità in µm/s, mediane)")
    print("=" * 100)
    for r in per_sample + per_split:
        name = r.get("campione", f"TOTALE {r['insieme'].upper()}")
        print(f" {r['insieme']:5s} {name:15s} {r['video']:3d} video {r['tracce']:5d} tracce | progressivi {r['progressivi %']:5.1f}% "
              f"(rapidi {r['progressivo rapido %']:5.1f}%) | non progr. {r['non progressivo %']:5.1f}% | immobili "
              f"{r['immobile %']:5.1f}% | VCL {r['VCL mediana']} VAP {r['VAP mediana']} VSL {r['VSL mediana']}")
    print(f"\n Pseudo-etichette: {n_pseudo[0]} box su {n_pseudo[1]} frame di {len(keys)} video "
          f"(in {out_dir / 'pseudo_labels'})")
    print(f"\n[OK] Output in {out_dir}")


if __name__ == "__main__":
    main()