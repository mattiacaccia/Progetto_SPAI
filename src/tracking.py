import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

TILE_PATTERN = re.compile(
    r"^(?P<sample>.+?)_(?P<ftl>FTL-\d{4}-\d+)_(?P<s2>.+?)_(?P<vid>V\d+)_frame_(?P<fr>\d+)_tile_(?P<x>\d+)_(?P<y>\d+)\.jpg$"
)
TILE_SIZE = 640
TILE_OVERLAP = 128          # come slice_dataset.py
CACHE_VERSION = "v2"        

DEFAULT_FPS = 30.0
DEFAULT_UM_PER_PX = 0.18548383342803432


# 1. Sequenze (video) del test set
def build_sequences(splits_file: Path, tiles_dir: Path, samples=None):
    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_samples = set(splits["test_samples"])
    if samples:
        test_samples &= set(samples)
    seqs = defaultdict(dict)
    for name, sample in splits["mapping_image_to_sample"].items():
        if sample not in test_samples:
            continue
        m = TILE_PATTERN.match(name)
        if not m:
            continue
        entry = seqs[(sample, m["vid"])].setdefault(int(m["fr"]), {"stem": name[: name.index("_tile_")], "tiles": [],
                                                                    "ftl": m["ftl"]})
        entry["tiles"].append((int(m["x"]), int(m["y"]), tiles_dir / name))
    return {k: dict(sorted(v.items())) for k, v in sorted(seqs.items())}


def tile_grid(W, H, tile=TILE_SIZE, overlap=TILE_OVERLAP):
    # Stessa griglia di slice_dataset.py
    stride, pos = tile - overlap, []
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            x_end, y_end = min(x + tile, W), min(y + tile, H)
            p = (max(0, x_end - tile), max(0, y_end - tile))
            if p not in pos:
                pos.append(p)
    return pos


def load_frame_tiles(entry, frames_dir: Path):

    import cv2
    full = None
    for ext in (".jpg", ".jpeg", ".png"):
        p = frames_dir / f"{entry['stem']}{ext}"
        if p.exists():
            full = cv2.imread(str(p))
            break
    if full is not None:
        H, W = full.shape[:2]
        tiles = [(x, y, full[y:y + TILE_SIZE, x:x + TILE_SIZE]) for x, y in tile_grid(W, H)]
        return tiles, full[:, :, ::-1]
    tiles = [(x, y, cv2.imread(str(p))) for x, y, p in entry["tiles"]]
    return tiles, None

# Detector Ultralytics (YOLOv8 / RT-DETR)
def load_detector(model_name, weights_path, device_str, det_conf):
    """detect(lista di tile BGR) -> lista di (box xyxy in pixel della tile [N,4], score [N])."""
    from ultralytics import YOLO, RTDETR
    model = YOLO(str(weights_path)) if model_name == "yolov8" else RTDETR(str(weights_path))

    def detect(images):
        res = model.predict(images, imgsz=TILE_SIZE, conf=det_conf, iou=0.6, max_det=300, device=device_str, verbose=False)
        return [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()) if r.boxes is not None and len(r.boxes)
                else (np.zeros((0, 4)), np.zeros(0)) for r in res]

    return detect

# Unione delle tile nel frame intero

def merge_nms(boxes, scores, iou_thr=0.5, ios_thr=0.5):

    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[rest]), np.maximum(y1[i], y1[rest])
        xx2, yy2 = np.minimum(x2[i], x2[rest]), np.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-12)
        ios = inter / (np.minimum(areas[i], areas[rest]) + 1e-12)
        order = rest[(iou < iou_thr) & (ios < ios_thr)]
    return np.array(keep, dtype=int)


def merge_tiles(tile_dets, offsets, iou_thr, ios_thr, frame_wh=None, edge_margin=2.0):
    #Riporta i box delle tile nel frame intero e rimuove i duplicati.
    W, H = frame_wh if frame_wh else (None, None)
    all_b, all_s = [], []
    for (b, s), (ox, oy) in zip(tile_dets, offsets):
        if not len(s):
            continue
        b = np.asarray(b, dtype=np.float64).copy()
        s = np.asarray(s, dtype=np.float64)
        if W is not None and edge_margin >= 0:
            inner_l, inner_t = ox > 0, oy > 0
            inner_r, inner_b = ox + TILE_SIZE < W, oy + TILE_SIZE < H
            cut = ((inner_l & (b[:, 0] <= edge_margin)) | (inner_t & (b[:, 1] <= edge_margin)) |
                   (inner_r & (b[:, 2] >= TILE_SIZE - edge_margin)) | (inner_b & (b[:, 3] >= TILE_SIZE - edge_margin)))
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


def detect_video(seq, detect, frames_dir, args):
    frame_dets, n_full = [], 0
    for frame, entry in seq.items():
        tiles, full = load_frame_tiles(entry, frames_dir)
        n_full += full is not None
        dets = detect([t for _, _, t in tiles])
        # il filtro sui bordi interni è sicuro solo con la griglia completa (frame intero); con le tile salvate alcune mancano
        wh = (full.shape[1], full.shape[0]) if full is not None else None
        boxes, scores = merge_tiles(dets, [(x, y) for x, y, _ in tiles], args.merge_iou, args.merge_ios, wh, args.edge_margin)
        frame_dets.append((frame, boxes, scores))
    return frame_dets, n_full


#  Video originali (30 fps)
VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv", ".m4v")


def find_video(videos_dir: Path, sample: str, vid: str, ftl: str = None):
    files = [p for p in videos_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    exact = [p for p in files if p.stem == f"{sample}_{vid}"]
    if exact:
        return exact[0]
    end = re.compile(rf"(^|[_\-\s]){re.escape(vid)}$", re.IGNORECASE)
    by_sample = [p for p in files if sample in p.stem and end.search(p.stem)]
    if len(by_sample) == 1:
        return by_sample[0]
    if ftl:
        by_ftl = [p for p in files if ftl in p.stem and end.search(p.stem)]
        if len(by_ftl) == 1:
            return by_ftl[0]
    return None


def _thumb(img_bgr):
    import cv2
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    g = cv2.resize(g, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def read_video_frames(video_path: Path, last_frame: int):
 
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video {video_path}")
    k = 0
    try:
        while k <= last_frame:
            ok, frame = cap.read()
            if not ok:
                break
            yield k, frame
            k += 1
    finally:
        cap.release()


def video_info(video_path: Path):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    info = {"fps": cap.get(cv2.CAP_PROP_FPS), "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "W": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "H": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return info


def check_alignment(video_path: Path, seq, frames_dir: Path, max_offset=5, n_check=8):
    import cv2
    ann = sorted(seq)
    pick = [ann[int(round(i))] for i in np.linspace(0, len(ann) - 1, min(n_check, len(ann)))]
    ref = {}
    for fr in pick:
        for ext in (".jpg", ".jpeg", ".png"):
            pth = frames_dir / f"{seq[fr]['stem']}{ext}"
            if pth.exists():
                ref[fr] = _thumb(cv2.imread(str(pth)))
                break
    if not ref:
        return None, {}, None
    need = set(fr + k for fr in ref for k in range(-max_offset, max_offset + 1))
    thumbs = {k: _thumb(f) for k, f in read_video_frames(video_path, max(need)) if k in need}
    err = {}
    for k in range(-max_offset, max_offset + 1):
        e = [np.abs(thumbs[fr + k] - t).mean() for fr, t in ref.items() if fr + k in thumbs]
        if len(e) == len(ref):
            err[k] = float(np.mean(e))
    if not err:
        return None, {}, None
    order = sorted(err, key=err.get)
    ratio = err[order[1]] / max(err[order[0]], 1e-9) if len(order) > 1 else None
    return order[0], err, ratio


def detect_video_file(video_path: Path, seq, detect, args, offset):

    ann = sorted(seq)
    first, last = max(0, ann[0] - args.video_margin), ann[-1] + args.video_margin
    frame_dets, bg = [], None
    for k, frame in read_video_frames(video_path, last + offset):
        fr = k - offset                       # indice nella numerazione delle annotazioni
        if fr < first:
            continue
        H, W = frame.shape[:2]
        if bg is None:
            bg = frame[:, :, ::-1].copy()
        tiles = [(x, y, frame[y:y + TILE_SIZE, x:x + TILE_SIZE]) for x, y in tile_grid(W, H)]
        dets = detect([t for _, _, t in tiles])
        boxes, scores = merge_tiles(dets, [(x, y) for x, y, _ in tiles], args.merge_iou, args.merge_ios, (W, H), args.edge_margin)
        frame_dets.append((fr, boxes, scores))
    return frame_dets, bg


def export_pseudo_labels(rows, casa, tag, out_dir: Path, annotated, W, H, min_len):
    # Pseudo-etichette dei frame NON annotati, solo dalle tracce usate per CASA
    keep = {c["track_id"] for c in casa}
    d = out_dir / "pseudo_labels"
    (d / tag).mkdir(parents=True, exist_ok=True)
    by_f = defaultdict(list)
    with open(d / f"{tag}_mot.txt", "w") as f:
        for fr, tid, x1, y1, x2, y2, sc in sorted(rows):
            if tid not in keep:
                continue
            f.write(f"{fr + 1},{tid},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{sc:.3f},-1,-1,-1\n")
            if fr not in annotated:
                by_f[fr].append((x1, y1, x2, y2))
    for fr, bxs in by_f.items():
        with open(d / tag / f"{tag}_frame_{fr:04d}.txt", "w") as f:
            for x1, y1, x2, y2 in bxs:
                f.write(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
    return len(by_f)


def make_gif_video(video_path: Path, rows, offset, out_path, first, last, step=3, scale=0.4, tail=15, fps_gif=10):
    import cv2
    from PIL import Image
    by_f = defaultdict(list)
    for fr, tid, x1, y1, x2, y2, _ in rows:
        by_f[fr].append((tid, x1, y1, x2, y2))
    hist = defaultdict(list)
    rng = np.random.default_rng(0)
    col = defaultdict(lambda: tuple(int(v) for v in rng.integers(40, 230, 3)))
    imgs = []
    for k, frame in read_video_frames(video_path, last + offset):
        fr = k - offset
        if fr < first:
            continue
        for tid, x1, y1, x2, y2 in by_f.get(fr, []):
            hist[tid].append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
        if (fr - first) % step:
            continue
        im = np.ascontiguousarray(frame[:, :, ::-1])
        for tid, x1, y1, x2, y2 in by_f.get(fr, []):
            c = col[tid]
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
            cv2.putText(im, str(tid), (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
            pts = hist[tid][-tail:]
            for a, b in zip(pts, pts[1:]):
                cv2.line(im, a, b, c, 2)
        cv2.putText(im, f"frame {fr}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(im, f"frame {fr}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
        imgs.append(Image.fromarray(cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)))
    if imgs:
        imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps_gif), loop=0)


# 4. ByteTrack (Ultralytics) con associazione per distanza dei centri

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


def make_tracker(args, px_per_frame_max):
    from ultralytics.trackers.byte_tracker import BYTETracker, STrack
    from ultralytics.trackers.basetrack import TrackState
    from ultralytics.trackers.utils import matching
    from ultralytics.utils import IterableSimpleNamespace

    cfg = IterableSimpleNamespace(
        tracker_type="bytetrack", track_high_thresh=args.track_high, track_low_thresh=args.track_low,
        new_track_thresh=args.new_track, track_buffer=args.track_buffer,
        match_thresh=args.match_thresh if args.assoc == "iou" else 1.0, fuse_score=args.assoc == "iou",
    )
    if args.assoc == "iou":           # ByteTrack originale (solo per confronto)
        return BYTETracker(cfg)

    class ObsSTrack(STrack):
        # STrack che ricorda posizione e frame reale dell'ultima detection associata.
        current_frame = 0

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
        # ByteTrack con associazione per distanza dei centri.

        track_class = ObsSTrack
        gap = 1

        def set_frame(self, frame_idx, gap):
            ObsSTrack.current_frame = frame_idx
            self.gap = gap

        def _center_cost(self, tracks, dets):
            if len(tracks) == 0 or len(dets) == 0:
                return np.zeros((len(tracks), len(dets)), dtype=np.float32)
            dc = np.array([d.xywh[:2] for d in dets], dtype=np.float32)
            pc = np.array([t.xywh[:2] for t in tracks], dtype=np.float32)
            last = np.array([t.last_obs for t in tracks], dtype=np.float32)
            dt = np.array([min(max(1, ObsSTrack.current_frame - t.last_obs_frame), args.max_gap_frames) for t in tracks],
                          dtype=np.float32)
            R = np.maximum(args.min_radius, px_per_frame_max * dt)[:, None]
            d_pred = np.linalg.norm(pc[:, None, :] - dc[None, :, :], axis=2)
            d_last = np.linalg.norm(last[:, None, :] - dc[None, :, :], axis=2)
            d = np.where(d_last <= R, np.minimum(d_pred, d_last), np.inf)
            return np.minimum(d / R, 1.01).astype(np.float32)     # > 1 = non associabile

        def get_dists(self, tracks, detections):
            return self._center_cost(tracks, detections)

        def multi_predict(self, tracks):
            for _ in range(self.gap):       # un passo del filtro di Kalman per ogni frame reale trascorso
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


def run_tracker(frame_dets, args, px_per_frame_max):
    tracker = make_tracker(args, px_per_frame_max)
    rows, prev = [], None
    for frame_idx, boxes, scores in frame_dets:
        gap = max(1, frame_idx - prev) if prev is not None else 1
        if hasattr(tracker, "set_frame"):
            tracker.set_frame(frame_idx, gap)
        prev = frame_idx
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        out = tracker.update(Dets(boxes, scores))
        for r in (out if len(out) else []):
            # box della detection associata in questo frame
            x1, y1, x2, y2 = boxes[int(r[7])] if len(r) > 7 and 0 <= int(r[7]) < len(boxes) else r[:4]
            rows.append((frame_idx, int(r[4]), float(x1), float(y1), float(x2), float(y2), float(r[5])))
    return rows

# CASA
MOTILITY_CLASSES = ["progressivo rapido", "progressivo lento", "non progressivo", "immobile"]


def motility_class(vcl, vap, args):
    if vcl < args.static_vcl:
        return "immobile"
    if vap < args.slow_vap:
        return "non progressivo"
    if vap < args.rapid_vap:
        return "progressivo lento"
    return "progressivo rapido"


def smooth_path(xy, w=5):
    # Media mobile centrata su w punti
    half, n = (w - 1) // 2, len(xy)
    out, full = np.empty_like(xy), np.zeros(n, dtype=bool)
    for i in range(n):
        h = min(half, i, n - 1 - i)
        out[i] = xy[i - h:i + h + 1].mean(axis=0)
        full[i] = h == half
    return out, full


def beat_cross_frequency(xy, sp, full, frames, fps):
    # BCF (Hz): numero di volte al secondo in cui il percorso puntuale attraversa il percorso medio.
    idx = np.where(full)[0]
    if len(idx) < 3:
        return None
    tang = np.gradient(sp, axis=0)
    dev = xy - sp
    side = np.sign(tang[:, 0] * dev[:, 1] - tang[:, 1] * dev[:, 0])[idx]
    side = side[side != 0]
    T = (frames[idx[-1]] - frames[idx[0]]) / fps
    if len(side) < 2 or T <= 0:
        return None
    return float((side[1:] != side[:-1]).sum() / T)


def compute_casa(rows, fps, um_per_px, min_len, args):
    # Parametri CASA per traccia
    tracks = defaultdict(list)
    for frame, tid, x1, y1, x2, y2, _ in rows:
        tracks[tid].append((frame, (x1 + x2) / 2.0, (y1 + y2) / 2.0))
    out, eps = [], 1e-9
    for tid, pts in sorted(tracks.items()):
        if len(pts) < min_len:
            continue
        pts.sort()
        frames = np.array([p[0] for p in pts], dtype=np.float64)
        xy = np.array([(p[1], p[2]) for p in pts], dtype=np.float64) * um_per_px
        T = (frames[-1] - frames[0]) / fps
        if T <= 0:
            continue
        vcl = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() / T
        vsl = np.linalg.norm(xy[-1] - xy[0]) / T
        sp, full = smooth_path(xy, args.vap_window)
        vap = min(np.linalg.norm(np.diff(sp, axis=0), axis=1).sum() / T, vcl)   # VSL <= VAP <= VCL
        dev = np.linalg.norm(xy - sp, axis=1)
        alh = float(dev[full].mean()) if full.any() else float(dev.mean())
        out.append({
            "track_id": tid, "n_points": len(pts), "frame_start": int(frames[0]), "frame_end": int(frames[-1]),
            "duration_s": round(float(T), 3), "VCL": round(float(vcl), 2), "VSL": round(float(vsl), 2),
            "VAP": round(float(vap), 2), "LIN_pct": round(float(vsl / (vcl + eps) * 100), 1),
            "STR_pct": round(float(vsl / (vap + eps) * 100), 1), "WOB_pct": round(float(vap / (vcl + eps) * 100), 1),
            "ALH": round(alh, 2),
            **({"BCF": (lambda b: round(b, 2) if b is not None else None)(beat_cross_frequency(xy, sp, full, frames, fps))}
               if args.source == "video" else {}),
            "classe": motility_class(vcl, vap, args),
        })
    return out


def proxy_stats(rows, frame_dets, min_len):

    lens = defaultdict(int)
    for _, tid, *_ in rows:
        lens[tid] += 1
    L = np.array(list(lens.values())) if lens else np.zeros(0)
    n_dets = np.array([len(s) for _, _, s in frame_dets]) if frame_dets else np.zeros(0)
    frames = [f for f, _, _ in frame_dets]
    gaps = np.diff(frames) if len(frames) > 1 else np.zeros(0)
    return {
        "n_frames": len(frame_dets),
        "mean_frame_gap": round(float(gaps.mean()), 2) if len(gaps) else 0,
        "max_frame_gap": int(gaps.max()) if len(gaps) else 0,
        "mean_dets_per_frame": round(float(n_dets.mean()), 2) if len(n_dets) else 0,
        "n_tracks": int(len(L)),
        "n_tracks_len_ge_min": int((L >= min_len).sum()) if len(L) else 0,
        "mean_track_len": round(float(L.mean()), 2) if len(L) else 0,
        "median_track_len": float(np.median(L)) if len(L) else 0,
        "max_track_len": int(L.max()) if len(L) else 0,
        "tracks_per_mean_det": round(float(len(L) / n_dets.mean()), 2) if len(n_dets) and n_dets.mean() > 0 else 0,
    }

# Accuratezza dei box sui frame annotati (senza identità)

def load_gt_fullframe(stem, labels_dir: Path, images_dir: Path):
    lbl = labels_dir / f"{stem}.txt"
    if not lbl.exists():
        return None
    size = None
    for ext in (".jpg", ".jpeg", ".png"):
        im = images_dir / f"{stem}{ext}"
        if im.exists():
            from PIL import Image
            with Image.open(im) as f:
                size = f.size
            break
    if size is None:
        return None
    W, H = size
    boxes = []
    for line in lbl.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5 and int(float(parts[0])) == 0:  # classe 0 = testa dello spermatozoo
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    return np.array(boxes, dtype=np.float64).reshape(-1, 4)


def _iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def count_tp(pred, gt, iou_thr=0.5):
    iou = _iou_matrix(pred, gt)
    if iou.size == 0:
        return 0, 0.0, 0.0
    pairs = sorted(((iou[i, j], i, j) for i in range(iou.shape[0]) for j in range(iou.shape[1]) if iou[i, j] >= iou_thr), reverse=True)
    used_p, used_g, tp, s_iou, s_d = set(), set(), 0, 0.0, 0.0
    for v, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            tp += 1
            s_iou += v
            s_d += float(np.hypot((pred[i, 0] + pred[i, 2] - gt[j, 0] - gt[j, 2]) / 2, (pred[i, 1] + pred[i, 3] - gt[j, 1] - gt[j, 3]) / 2))
    return tp, s_iou, s_d


def evaluate_vs_gt(seq, frame_dets, rows, labels_dir, images_dir, det_thr, iou_thr=0.5):
    by_frame = defaultdict(list)
    for fr, _tid, x1, y1, x2, y2, _s in rows:
        by_frame[fr].append([x1, y1, x2, y2])
    keys = ("frames", "gt", "trk_tp", "trk_n", "trk_iou", "trk_dist", "det_tp", "det_n", "det_iou", "det_dist")
    t = dict.fromkeys(keys, 0)
    for fr, boxes, scores in frame_dets:
        if fr not in seq:                     # a 30 fps solo i frame annotati hanno il ground truth
            continue
        gt = load_gt_fullframe(seq[fr]["stem"], labels_dir, images_dir)
        if gt is None:
            continue
        trk = np.array(by_frame.get(fr, []), dtype=np.float64).reshape(-1, 4)
        det = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)[np.asarray(scores) >= det_thr]
        t["frames"] += 1
        t["gt"] += len(gt)
        for k, b in (("trk", trk), ("det", det)):
            tp, s_iou, s_d = count_tp(b, gt, iou_thr)
            t[f"{k}_n"] += len(b)
            t[f"{k}_tp"] += tp
            t[f"{k}_iou"] += s_iou
            t[f"{k}_dist"] += s_d
    return t if t["frames"] else None


def gt_metrics(tp, n_pred, n_gt, s_iou=0.0, s_dist=0.0, um_per_px=DEFAULT_UM_PER_PX):
    fp, fn = n_pred - tp, n_gt - tp
    p, r = tp / max(n_pred, 1), tp / max(n_gt, 1)
    return {"P": round(p, 4), "R": round(r, 4), "F1": round(2 * p * r / max(p + r, 1e-12), 4),
            "MOTA_senza_IDSW": round(1 - (fn + fp) / max(n_gt, 1), 4),
            "MOTP_IoU": round(s_iou / tp, 4) if tp else None,
            "errore_centro_um": round(s_dist / tp * um_per_px, 3) if tp else None}


def load_identity_gt(path: Path):
    out = defaultdict(lambda: defaultdict(list))
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("track_code") and r.get("head_x_px"):
                vid = r["video"].replace(".avi", "")
                out[vid][r["track_code"]].append((int(r["frame"]), float(r["head_x_px"]), float(r["head_y_px"])))
    return {v: {k: sorted(set(p)) for k, p in t.items() if len(set(p)) >= 2} for v, t in out.items()}


def identity_check(rows, gt_tracks, tol_px=15.0):
    # Confronto con le identità annotate (tracking.csv): ogni punto annotato è assegnato alla traccia ByteTrack
    # il cui centro è più vicino.

    from scipy.optimize import linear_sum_assignment
    by_f = defaultdict(list)
    for fr, tid, x1, y1, x2, y2, _ in rows:
        by_f[fr].append((tid, (x1 + x2) / 2, (y1 + y2) / 2))
    res = {"tracce_annotate": 0, "punti_annotati": 0, "punti_coperti": 0, "cambi_id": 0}
    counts = defaultdict(lambda: defaultdict(int))
    for g, pts in enumerate(gt_tracks.values()):
        seq = []
        for fr, x, y in pts:
            c = [(np.hypot(px - x, py - y), tid) for tid, px, py in by_f.get(fr, [])]
            c = [t for t in c if t[0] <= tol_px]
            res["punti_annotati"] += 1
            if c:
                res["punti_coperti"] += 1
                seq.append(min(c)[1])
                counts[g][min(c)[1]] += 1
        res["tracce_annotate"] += 1
        res["cambi_id"] += sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    pred_ids = sorted({p for d in counts.values() for p in d})
    idtp = 0
    if pred_ids:
        M = np.zeros((res["tracce_annotate"], len(pred_ids)))
        for g, d in counts.items():
            for p, n in d.items():
                M[g, pred_ids.index(p)] = n
        r_i, c_i = linear_sum_assignment(-M)
        idtp = int(M[r_i, c_i].sum())
    res["IDTP"] = idtp
    res["IDR"] = round(idtp / max(res["punti_annotati"], 1), 4)
    return res



# Grafici

CLASS_COLORS = {"progressivo rapido": "#138a36", "progressivo lento": "#8cc63f", "non progressivo": "#e8590c",
                "immobile": "#2f5fb3"}


def plot_trajectories(rows, casa, background, title, out_path, min_len, units_note, vap_window=5):
    # Linea spessa = percorso medio (lo stesso usato per la VAP); linea sottile tratteggiata = percorso puntuale
    # (VCL), con un punto per ogni frame campionato; pallino = inizio con l'ID della traccia, freccia = fine.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cls = {c["track_id"]: c["classe"] for c in casa}
    tracks = defaultdict(list)
    for frame, tid, x1, y1, x2, y2, _ in rows:
        tracks[tid].append((frame, (x1 + x2) / 2, (y1 + y2) / 2))
    H, W = (background.shape[:2] if background is not None else (720, 1280))
    fig, ax = plt.subplots(figsize=(15.5, 13 * H / W + 0.8))
    if background is not None:
        ax.imshow(background, cmap="gray", alpha=0.5)      # già RGB (load_frame_tiles)
    counts = defaultdict(int)
    for tid, pts in tracks.items():
        if len(pts) < min_len or tid not in cls:
            continue
        pts.sort()
        a = np.array([(p[1], p[2]) for p in pts])
        sm, _ = smooth_path(a, vap_window)
        c = CLASS_COLORS[cls[tid]]
        counts[cls[tid]] += 1
        ax.plot(a[:, 0], a[:, 1], "--", color=c, linewidth=0.8, alpha=0.7)
        ax.plot(a[:, 0], a[:, 1], ".", color=c, markersize=3.5 if len(a) <= 40 else 1.5, alpha=0.9)
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
                Line2D([0], [0], color="#555555", lw=0.8, ls="--", marker=".", label="percorso puntuale (VCL)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=False,
              title=f"tracce ≥ {min_len} punti")
    fig.text(0.01, 0.005, units_note, fontsize=8, color="#555555")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def make_gif(seq, rows, frames_dir, out_path, fps_gif=3, scale=0.5, tail=6):
 
    import cv2
    from PIL import Image
    by_f = defaultdict(list)
    for fr, tid, x1, y1, x2, y2, _ in rows:
        by_f[fr].append((tid, x1, y1, x2, y2))
    hist = defaultdict(list)
    rng = np.random.default_rng(0)
    col = defaultdict(lambda: tuple(int(v) for v in rng.integers(40, 230, 3)))
    imgs = []
    for fr in sorted(seq):
        _, full = load_frame_tiles(seq[fr], frames_dir)
        if full is None:
            continue
        im = np.ascontiguousarray(full) if full.ndim == 3 else cv2.cvtColor(full, cv2.COLOR_GRAY2RGB)   # già RGB
        for tid, x1, y1, x2, y2 in by_f.get(fr, []):
            hist[tid].append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
            c = col[tid]
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
            cv2.putText(im, str(tid), (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
            pts = hist[tid][-tail:]
            for a, b in zip(pts, pts[1:]):
                cv2.line(im, a, b, c, 2)
        cv2.putText(im, f"frame {fr}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(im, f"frame {fr}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        imgs.append(Image.fromarray(im))
    if imgs:
        imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps_gif), loop=0)


def stitch_tiles(tiles, W=1280, H=720):
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for x, y, t in tiles:
        canvas[y:y + t.shape[0], x:x + t.shape[1]] = t[:, :, ::-1]
    return canvas


# Main

def write_dicts(path, dict_rows):
    if not dict_rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict_rows[0].keys()))
        w.writeheader()
        w.writerows(dict_rows)


def main():
    ap = argparse.ArgumentParser(description="Tracking ByteTrack (Ultralytics) + CASA sulle detection di YOLOv8 / RT-DETR")
    ap.add_argument("--model", required=True, choices=["yolov8", "rtdetr"])
    ap.add_argument("--regime", required=True, choices=["regime_a", "regime_b"])
    ap.add_argument("--weights", default=None, help="Default: runs/train/<model>_<regime>/weights/best.pt")
    ap.add_argument("--frames-dir", default="data/processed_FertyLab/images", help="Frame interi 1280x720 (prima dello slicing)")
    ap.add_argument("--source", choices=["frames", "video"], default="frames",
                    help="frames = solo i frame annotati (~6 Hz); video = tutti i frame dei video originali (30 fps)")
    ap.add_argument("--videos-dir", default=None, help="Cartella dei video originali (cercati anche nelle sottocartelle)")
    ap.add_argument("--frame-offset", type=int, default=None,
                    help="Forza lo scarto tra frame del video e numerazione delle annotazioni (default: stimato e verificato)")
    ap.add_argument("--video-margin", type=int, default=0, help="Frame extra prima del primo e dopo l'ultimo frame annotato")
    ap.add_argument("--check-only", action="store_true", help="Solo verifica dei video (file trovato, fps, risoluzione, allineamento)")
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS, help="FPS di acquisizione (manifest FertyLab: 30)")
    ap.add_argument("--um-per-px", type=float, default=DEFAULT_UM_PER_PX, help="Calibrazione (manifest FertyLab: 0.18548)")
    # detection
    ap.add_argument("--det-conf", type=float, default=0.1, help="Confidenza minima delle detection passate al tracker")
    ap.add_argument("--merge-iou", type=float, default=0.5, help="Unione tile: soglia IoU")
    ap.add_argument("--merge-ios", type=float, default=1.01, help="Unione tile: soglia intersezione/area del box più piccolo (>1 = disattivata)")
    ap.add_argument("--edge-margin", type=float, default=2.0, help="Scarta i box a meno di N px da un bordo interno della tile (-1 = disattivato)")
    # tracking
    ap.add_argument("--assoc", choices=["distance", "iou"], default="distance", help="Costo di associazione (iou = ByteTrack originale)")
    ap.add_argument("--max-speed", type=float, default=60.0, help="Velocità massima della testa (µm/s) per il raggio di associazione "
                    "(60 ≈ 95° percentile delle velocità annotate in tracking.csv)")
    ap.add_argument("--min-radius", type=float, default=20.0, help="Raggio minimo di associazione (px)")
    ap.add_argument("--max-gap-frames", type=int, default=10, help="Il raggio cresce con i frame dall'ultima detection fino a questo limite")
    ap.add_argument("--track-high", type=float, default=0.3)
    ap.add_argument("--track-low", type=float, default=0.1)
    ap.add_argument("--new-track", type=float, default=0.3)
    ap.add_argument("--track-buffer", type=int, default=None,
                    help="Frame elaborati per cui una traccia persa resta recuperabile (default: 3 con frames, 15 con video)")
    ap.add_argument("--match-thresh", type=float, default=0.8, help="Solo per --assoc iou")
    # CASA
    ap.add_argument("--min-track-len", type=int, default=None, help="Punti minimi per CASA (default: 5 con frames, 30 = 1 s con video)")
    ap.add_argument("--vap-window", type=int, default=None, help="Punti della media mobile per la VAP (default: 5 con frames, 7 con video)")
    ap.add_argument("--static-vcl", type=float, default=5.0, help="VCL sotto cui la testa è considerata ferma = immobile (µm/s)")
    ap.add_argument("--slow-vap", type=float, default=5.0, help="VAP minima per 'progressivo lento' (OMS 2021: 5 µm/s)")
    ap.add_argument("--rapid-vap", type=float, default=25.0, help="VAP minima per 'progressivo rapido' (OMS 2021: 25 µm/s)")
    # esecuzione
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--plot-videos", type=int, default=-1, help="Quanti video disegnare (-1 = tutti)")
    ap.add_argument("--gif", action="store_true", help="Salva anche una GIF per video con box, ID e scia delle tracce")
    ap.add_argument("--export-pseudo", action="store_true",
                    help="Solo con --source video: salva le pseudo-etichette dei frame non annotati (MOTChallenge + YOLO)")
    ap.add_argument("--use-cache", action="store_true", help="Riusa le detection salvate (per provare altri parametri del tracker)")
    ap.add_argument("--detect-only", action="store_true", help="Solo object detection (salvata in cache); il tracking si lancia dopo con --use-cache")
    ap.add_argument("--gt-labels", default="data/processed_FertyLab/labels", help="Label YOLO dei frame interi per la valutazione")
    ap.add_argument("--id-gt", default=None, help="tracking.csv del dataset FertyLab (csv/tracking.csv) per il controllo delle identità")
    ap.add_argument("--device", default=None, help="mps | cpu (default: mps se disponibile)")
    args = ap.parse_args()
    video_mode = args.source == "video"
    for k, v_frames, v_video in (("track_buffer", 3, 15), ("min_track_len", 5, 30), ("vap_window", 5, 7)):
        if getattr(args, k) is None:
            setattr(args, k, v_video if video_mode else v_frames)
    if video_mode and not args.videos_dir:
        ap.error("--source video richiede --videos-dir")

    if args.device:
        device_str = args.device
    else:
        import torch
        device_str = "mps" if torch.backends.mps.is_available() else "cpu"
    px_per_frame_max = args.max_speed / args.um_per_px / args.fps
    print(f"[INFO] Tracking con {args.model.upper()} ({args.regime}) | associazione: {args.assoc} | "
          f"v_max {args.max_speed:.0f} µm/s = {px_per_frame_max:.1f} px/frame | device {device_str}")

    weights = Path(args.weights) if args.weights else Path(f"runs/train/{args.model}_{args.regime}/weights/best.pt")
    out_dir = Path("runs/tracking_casa") / (f"{args.model}_{args.regime}" + ("_video" if video_mode else ""))
    for sub in ("tracks", "det_cache", "trajectories"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    base = project_root / "data" / "sliced_processed_FertyLab"
    seqs = build_sequences(base / "splits.json", base / "images")
    if not seqs:
        raise RuntimeError("Nessuna sequenza di test trovata: controlla splits.json e il nome delle tile.")
    keys = list(seqs)[: args.max_videos] if args.max_videos else list(seqs)
    print(f"[INFO] {len(keys)} video di test")

    frames_dir, labels_dir = Path(args.frames_dir), Path(args.gt_labels)
    if not frames_dir.exists():
        print(f"[AVVISO] {frames_dir} non trovata: uso le tile salvate (solo quelle con almeno un'annotazione).")
    gt_available = labels_dir.exists() and frames_dir.exists()

    id_gt = load_identity_gt(Path(args.id_gt)) if args.id_gt else {}
    videos_dir = Path(args.videos_dir) if video_mode else None
    align_rows = []
    detect = None
    proxy_rows, casa_rows, gt_rows, id_rows = [], [], [], []
    for n_vid, (sample, vid) in enumerate(keys):
        seq = seqs[(sample, vid)]
        tag = f"{sample}_{vid}"
        cache = out_dir / "det_cache" / f"{tag}_{CACHE_VERSION}.npz"
        video_path, offset, video_bg = None, 0, None

        if video_mode:
            video_path = find_video(videos_dir, sample, vid, next(iter(seq.values())).get("ftl"))
            if video_path is None:
                print(f"[{n_vid + 1}/{len(keys)}] {tag}: video NON trovato in {videos_dir}, salto.")
                align_rows.append({"sample": sample, "video": vid, "file": "", "esito": "video non trovato"})
                continue
            info = video_info(video_path)
            best, err, ratio = check_alignment(video_path, seq, frames_dir)
            if args.frame_offset is not None:
                offset, esito = args.frame_offset, "offset impostato a mano"
            elif best is None:
                offset, esito = 0, "allineamento NON verificabile (frame annotati non trovati)"
            elif ratio is not None and ratio < 1.15:
                offset, esito = 0, f"allineamento INCERTO (scena quasi ferma, rapporto {ratio:.2f}): uso offset 0"
            else:
                offset, esito = best, ("allineato" if best == 0 else f"SFASATO di {best} frame: corretto")
            align_rows.append({"sample": sample, "video": vid, "file": video_path.name, "fps": round(info["fps"], 2),
                               "frame_video": info["n_frames"], "risoluzione": f"{info['W']}x{info['H']}",
                               "ultimo_frame_annotato": max(seq), "offset_stimato": best, "offset_usato": offset,
                               "rapporto_2o_1o": round(ratio, 2) if ratio else None, "esito": esito})
            warn = []
            if abs(info["fps"] - args.fps) > 0.5:
                warn.append(f"fps del video {info['fps']:.2f} diverso da --fps {args.fps:g}")
            if (info["W"], info["H"]) != (1280, 720):
                warn.append(f"risoluzione {info['W']}x{info['H']} diversa da 1280x720")
            if info["n_frames"] and info["n_frames"] <= max(seq) + offset:
                warn.append(f"il video ha {info['n_frames']} frame ma l'ultimo annotato è {max(seq)}")
            print(f"[{n_vid + 1}/{len(keys)}] {tag}: {video_path.name} | {info['n_frames']} frame, {info['fps']:.1f} fps, "
                  f"{info['W']}x{info['H']} | {esito}" + ("".join(f"\n      [AVVISO] {w}" for w in warn)))
            if args.check_only:
                continue

        t0 = time.time()
        if args.use_cache and cache.exists():
            z = np.load(cache)
            frame_dets = [(int(f), z[f"b{f}"], z[f"s{f}"]) for f in z["frames"]]
            n_full = int(z["n_full"]) if "n_full" in z else -1
        else:
            if args.use_cache:
                print(f"[AVVISO] cache {cache.name} assente: eseguo la detection.")
            if detect is None:
                if not weights.exists():
                    raise FileNotFoundError(f"Pesi non trovati: {weights}")
                detect = load_detector(args.model, weights, device_str, args.det_conf)
            if video_mode:
                frame_dets, video_bg = detect_video_file(video_path, seq, detect, args, offset)
                n_full = len(frame_dets)
            else:
                frame_dets, n_full = detect_video(seq, detect, frames_dir, args)
            np.savez_compressed(cache, frames=np.array([f for f, _, _ in frame_dets]), n_full=n_full,
                                **{f"b{f}": b for f, b, _ in frame_dets}, **{f"s{f}": s for f, _, s in frame_dets})
        t_det = time.time() - t0
        if args.detect_only:
            src = ("video" if video_mode else "frame interi") if n_full == len(frame_dets) else f"{n_full}/{len(frame_dets)} frame interi"
            print(f"[{n_vid + 1}/{len(keys)}] {tag}: {len(frame_dets)} frame ({src}), "
                  f"{np.mean([len(s) for _, _, s in frame_dets]):.1f} detection/frame | {t_det:.0f}s")
            continue

        rows = run_tracker(frame_dets, args, px_per_frame_max)
        with open(out_dir / "tracks" / f"{tag}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "score"])
            w.writerows(rows)

        st = proxy_stats(rows, frame_dets, args.min_track_len)
        proxy_rows.append({"sample": sample, "video": vid, **st})
        casa = compute_casa(rows, args.fps, args.um_per_px, args.min_track_len, args)
        casa_rows += [{"sample": sample, "video": vid, **c} for c in casa]
        if gt_available:
            g = evaluate_vs_gt(seq, frame_dets, rows, labels_dir, frames_dir, args.track_high)
            if g:
                gt_rows.append({"sample": sample, "video": vid, **g})
        if id_gt.get(tag):
            id_rows.append({"sample": sample, "video": vid, **identity_check(rows, id_gt[tag])})
        n_pseudo = None
        if video_mode and args.export_pseudo:
            n_pseudo = export_pseudo_labels(rows, casa, tag, out_dir, set(seq), 1280, 720, args.min_track_len)
        n_cls = defaultdict(int)
        for c in casa:
            n_cls[c["classe"]] += 1
        prog = n_cls["progressivo rapido"] + n_cls["progressivo lento"]
        print(f"[{n_vid + 1}/{len(keys)}] {tag}: {st['mean_dets_per_frame']} det/frame | {st['n_tracks']} tracce, "
              f"{st['n_tracks_len_ge_min']} con >= {args.min_track_len} punti (lunghezza mediana {st['median_track_len']:.0f}) | "
              f"progressivi {prog} (rapidi {n_cls['progressivo rapido']}), non progressivi {n_cls['non progressivo']}, immobili {n_cls['immobile']}"
              + (f" | pseudo-etichette su {n_pseudo} frame non annotati" if n_pseudo is not None else ""))

        if args.plot_videos < 0 or n_vid < args.plot_videos:
            first = next(iter(seq))
            tiles, full = load_frame_tiles(seq[first], frames_dir)
            bg = full if full is not None else stitch_tiles(tiles)
            what = "frame consecutivi (30 fps)" if video_mode else "frame campionati"
            span = (frame_dets[-1][0] - frame_dets[0][0]) / args.fps if frame_dets else 0
            plot_trajectories(rows, casa, bg, f"{tag} - {args.model.upper()} {args.regime} - ByteTrack ({args.assoc}) - "
                              f"{st['n_frames']} {what} in {span:.1f} s",
                              out_dir / "trajectories" / f"{tag}.png", args.min_track_len,
                              f"Sfondo: primo frame (gli spermatozoi mobili si trovano lì solo all'inizio, al pallino con l'ID). "
                              f"Classi OMS 2021: immobile VCL < {args.static_vcl:g}; non progressivo VAP < {args.slow_vap:g}; "
                              f"progressivo lento VAP < {args.rapid_vap:g}; progressivo rapido VAP >= {args.rapid_vap:g} µm/s.",
                              args.vap_window)
            if args.gif:
                if video_mode:
                    make_gif_video(video_path, rows, offset, out_dir / "trajectories" / f"{tag}.gif",
                                   frame_dets[0][0], frame_dets[-1][0])
                else:
                    make_gif(seq, rows, frames_dir, out_dir / "trajectories" / f"{tag}.gif")

    if align_rows:
        write_dicts(out_dir / "video_alignment.csv", align_rows)
        print(f"\n[INFO] Verifica dei video in {out_dir / 'video_alignment.csv'}")
    if args.check_only:
        return
    if args.detect_only:
        print(f"\n[OK] Detection salvate in {out_dir / 'det_cache'}. Ora: stesso comando con --use-cache al posto di --detect-only.")
        return

    write_dicts(out_dir / "tracking_proxy_report.csv", proxy_rows)
    write_dicts(out_dir / "casa_kinematics_report.csv", casa_rows)

    print("\n" + "=" * 86)
    print(" TRACCE (statistiche senza identità di riferimento)")
    print("=" * 86)
    for key in ["mean_dets_per_frame", "n_tracks", "n_tracks_len_ge_min", "median_track_len", "tracks_per_mean_det"]:
        vals = [r[key] for r in proxy_rows]
        print(f" - {key}: media {np.mean(vals):.2f} (min {np.min(vals):.2f}, max {np.max(vals):.2f})")

    summary = []
    if casa_rows:
        groups = [("tutte", casa_rows)] + [(k, [r for r in casa_rows if r["classe"] == k]) for k in MOTILITY_CLASSES]
        groups.insert(1, ("progressivi (rapidi + lenti)", [r for r in casa_rows if r["classe"].startswith("progressivo")]))
        print(f"\n CASA ({len(casa_rows)} tracce con >= {args.min_track_len} punti, µm/s; ALH in µm)")
        for name, grp in groups:
            line = {"gruppo": name, "n_tracce": len(grp), "percentuale": round(100 * len(grp) / len(casa_rows), 1)}
            for k in ["VCL", "VSL", "VAP", "LIN_pct", "STR_pct", "WOB_pct", "ALH"] + (["BCF"] if video_mode else []):
                v = np.array([r[k] for r in grp if r[k] is not None], dtype=float)
                line[f"{k}_media"] = round(float(v.mean()), 2) if len(v) else None
                line[f"{k}_mediana"] = round(float(np.median(v)), 2) if len(v) else None
                line[f"{k}_std"] = round(float(v.std()), 2) if len(v) else None
            summary.append(line)
            print(f" - {name:28s} n={len(grp):4d} ({line['percentuale']:5.1f}%) | VCL {line['VCL_mediana']} | VSL {line['VSL_mediana']} | "
                  f"VAP {line['VAP_mediana']} | LIN {line['LIN_pct_mediana']}% | ALH {line['ALH_mediana']} (mediane)")
        write_dicts(out_dir / "casa_summary.csv", summary)
    else:
        print("\n Nessuna traccia abbastanza lunga per il calcolo CASA.")

    if gt_rows:
        tot = {k: sum(r[k] for r in gt_rows) for k in gt_rows[0] if k not in ("sample", "video")}
        report = []
        for r in gt_rows + [{"sample": "TOTALE", "video": "", **tot}]:
            for kind, k in (("detection (score>=track_high)", "det"), ("tracce ByteTrack", "trk")):
                report.append({"sample": r["sample"], "video": r["video"], "tipo": kind, "frame": r["frames"], "GT": r["gt"],
                               "box_predetti": r[f"{k}_n"], "TP": r[f"{k}_tp"],
                               **gt_metrics(r[f"{k}_tp"], r[f"{k}_n"], r["gt"], r[f"{k}_iou"], r[f"{k}_dist"], args.um_per_px)})
        write_dicts(out_dir / "tracking_vs_gt_report.csv", report)
        print("\n" + "=" * 86)
        print(f" BOX SUI FRAME ANNOTATI (IoU >= 0.5, {tot['frames']} frame, {tot['gt']} spermatozoi annotati) - senza identità")
        print("=" * 86)
        for name, k in (("detection (score >= track_high)", "det"), ("tracce ByteTrack", "trk")):
            m = gt_metrics(tot[f"{k}_tp"], tot[f"{k}_n"], tot["gt"], tot[f"{k}_iou"], tot[f"{k}_dist"], args.um_per_px)
            print(f" - {name}: P {m['P']:.3f} | R {m['R']:.3f} | F1 {m['F1']:.3f} | MOTA senza IDSW {m['MOTA_senza_IDSW']:.3f} | "
                  f"MOTP (IoU) {m['MOTP_IoU']} | errore centro {m['errore_centro_um']} µm")
    if id_rows:
        write_dicts(out_dir / "identity_check.csv", id_rows)
        tot = {k: sum(r[k] for r in id_rows) for k in ("tracce_annotate", "punti_annotati", "punti_coperti", "cambi_id", "IDTP")}
        print(f"\n CONTROLLO IDENTITÀ (tracking.csv, {len(id_rows)} video): {tot['tracce_annotate']} tracce annotate, "
              f"{tot['punti_coperti']}/{tot['punti_annotati']} punti coperti da una traccia, {tot['cambi_id']} cambi di ID, "
              f"IDR {tot['IDTP'] / max(tot['punti_annotati'], 1):.3f}")
    elif args.id_gt:
        print("\n CONTROLLO IDENTITÀ: nessuna traccia annotata con coordinate nei video di test elaborati.")
    print(f"\n[OK] Output in: {out_dir}  (grafici in {out_dir / 'trajectories'})")


if __name__ == "__main__":
    main()