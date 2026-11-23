
import csv
import io
import json
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

def _project_root():
    here = Path(__file__).resolve().parent
    for d in (here, *here.parents):
        if (d / "pyproject.toml").exists() or (d / "data").is_dir() and (d / "src").is_dir():
            return d
    return here


ROOT = _project_root()
sys.path.insert(0, str(ROOT))


# Configurazione
MODELS = {"yolov8": "YOLOv8", "rtdetr": "RT-DETR", "detrvit": "DETR ViT-based", "todcnn": "TOD-CNN"}
REGIMES = {"regime_a": "Scenario A (SVIA + FertyLab)", "regime_b": "Scenario B (solo FertyLab)"}

DET_TEST_SAMPLES = ["SEM-0002", "SEM-0008"]
TRACK_TEST_SAMPLES = ["SEM-0002", "SEM-0005", "SEM-0006", "SEM-0008", "SEM-0009", "SEM-0010",
                      "SEM-0011", "SEM-0014", "SEM-0015", "SEM-0016", "SEM-0017"]
TRACK_MODEL, TRACK_REGIME = "yolov8", "regime_a"

FERTY = ROOT / "data" / "sliced_processed_FertyLab"
VIDEOS_DIR = ROOT / "data" / "spai_video"
TRACK_OUT = ROOT / "runs" / "tracking_video" / f"{TRACK_MODEL}_{TRACK_REGIME}"

GT_COLOR, PRED_COLOR = (45, 91, 215), (220, 38, 38)          
CLASS_COLORS = {"progressivo rapido": "#138a36", "progressivo lento": "#8cc63f",
                "non progressivo": "#e8590c", "immobile": "#2f5fb3"}
SHORT_COLOR = "#9aa5ad"                                      # tracce < 30 punti non usate per il CASA

# stessi valori di default di src/tracking.py
TRACK_ARGS = SimpleNamespace(det_conf=0.1, merge_iou=0.5, merge_ios=1.01, edge_margin=2.0, min_track_len=30,
                             vap_window=7, static_vcl=5.0, slow_vap=5.0, rapid_vap=25.0)
THESIS_PARAMS = {"max_speed": 86.8, "min_radius": 10.0, "max_gap_frames": 30, "track_high": 0.3,
                 "track_low": 0.1, "new_track": 0.3, "track_buffer": 20}

st.set_page_config(page_title="Demo Progetto SPAI", page_icon="🔬", layout="wide")


# Utilità comuni

def get_device():
    try:
        import torch
        return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def find_weights(model, regime):
    # best.pt nelle cartelle usate dagli script di training 
    for base in ("runs/train", "runs/detect/runs/train", "runs/detect/train"):
        p = ROOT / base / f"{model}_{regime}" / "weights" / "best.pt"
        if p.exists():
            return p
    return None


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def read_csv_dicts(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# OBJECT DETECTION
@st.cache_data(show_spinner=False)
def list_test_tiles():
    splits_file = FERTY / "splits.json"
    if not splits_file.exists():
        return None
    with open(splits_file, encoding="utf-8") as f:
        mapping = json.load(f)["mapping_image_to_sample"]
    out = {s: [] for s in DET_TEST_SAMPLES}
    img_dir = FERTY / "images"
    for p in sorted(img_dir.glob("*.*")):
        s = mapping.get(p.name)
        if s in out and not p.name.startswith("."):
            out[s].append(p.name)
    return out


def load_gt(img_name, W, H):
    lbl = FERTY / "labels" / (Path(img_name).stem + ".txt")
    boxes = []
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            v = line.split()
            if len(v) >= 5:
                cx, cy, w, h = map(float, v[1:5])
                boxes.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    return np.array(boxes, dtype=np.float64).reshape(-1, 4)


@st.cache_resource(show_spinner=False)
def load_det_model(model, weights_path):
    device = get_device()
    if model in ("yolov8", "rtdetr"):
        from ultralytics import YOLO, RTDETR
        return (YOLO if model == "yolov8" else RTDETR)(weights_path), device
    import torch
    ckpt = torch.load(weights_path, map_location=device)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if model == "detrvit":
        from src.models.vit_wrapper import DETRViTWrapper
        arch = ckpt.get("arch", {}) if isinstance(ckpt, dict) else {}
        net = DETRViTWrapper(num_classes=1, load_pretrained=False, **arch)
    else:
        from src.models.tod_cnn_wrapper import TODCNNWrapper
        net = TODCNNWrapper(num_classes=1)
    net.load_state_dict(sd)
    return net.to(device).eval(), device


def _decode_detr(outputs):
    import torch
    probs = outputs.logits[0].softmax(-1)[..., :-1]
    scores = probs.max(-1).values
    cx, cy, w, h = outputs.pred_boxes[0].unbind(-1)
    xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1).clamp(0, 1)
    return xyxy.float().cpu().numpy(), scores.float().cpu().numpy()


def _decode_todcnn(pred_map, max_cand=300, nms_iou=0.5, max_det=100):
    import torch
    from torchvision.ops import nms
    pred_map = pred_map.detach().float().cpu()
    conf = torch.sigmoid(pred_map[4]).reshape(-1)
    top_conf, idx = torch.topk(conf, min(max_cand, conf.numel()))
    keep = top_conf > 0.001
    top_conf, idx = top_conf[keep], idx[keep]
    if idx.numel() == 0:
        return np.zeros((0, 4)), np.zeros(0)
    Wm = pred_map.shape[2]
    ys, xs = idx // Wm, idx % Wm
    cx, cy, w, h = (pred_map[c, ys, xs] for c in range(4))
    xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1).clamp(0, 1)
    k = nms(xyxy, top_conf, nms_iou)[:max_det]
    return xyxy[k].numpy(), top_conf[k].numpy()


def predict(model, net, device, image, conf):
    W, H = image.size
    if model in ("yolov8", "rtdetr"):
        r = net.predict(image, imgsz=640, conf=conf, device=device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0)
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
    import torch
    from torchvision import transforms
    size = getattr(net, "arch", {}).get("img_size", 640) if model == "detrvit" else 640
    x = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])(image.convert("RGB"))
    with torch.no_grad():
        out = net(x.unsqueeze(0).to(device))
    boxes, scores = _decode_detr(out) if model == "detrvit" else _decode_todcnn(out[0])
    keep = scores >= conf
    return boxes[keep] * np.array([W, H, W, H]), scores[keep]


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def tile_metrics(pred, scores, gt, thr=0.5):
    iou = iou_matrix(pred[np.argsort(-scores)], gt)
    used, tp = set(), 0
    for i in range(len(iou)):
        cand = [j for j in np.argsort(-iou[i]) if iou[i, j] >= thr and j not in used]
        if cand:
            used.add(cand[0])
            tp += 1
    p = tp / len(pred) if len(pred) else 0.0
    r = tp / len(gt) if len(gt) else 0.0
    return tp, p, r


def draw_boxes(image, gt=None, pred=None, width=2):
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    for b in (gt if gt is not None else []):
        d.rectangle(list(map(float, b)), outline=GT_COLOR, width=width)
    for b in (pred if pred is not None else []):
        d.rectangle(list(map(float, b)), outline=PRED_COLOR, width=width)
    return img


def official_test_metrics():
    rows = []
    for m, mname in MODELS.items():
        for r, rname in REGIMES.items():
            f = ROOT / "runs" / "test" / f"{m}_{r}" / "test_results.csv"
            if f.exists():
                d = read_csv_dicts(f)[0]
                rows.append({"Modello": mname, "Scenario": rname.split(" (")[0],
                             "mAP@50": round(float(d["metrics/mAP50(B)"]), 3),
                             "mAP@50-95": round(float(d["metrics/mAP50-95(B)"]), 3),
                             "Precision": round(float(d["metrics/precision(B)"]), 3),
                             "Recall": round(float(d["metrics/recall(B)"]), 3),
                             "F1": round(float(d["metrics/f1_score(B)"]), 3)})
    return rows


def detection_page():
    st.subheader("Object Detection sul test set (SEM-0002, SEM-0008)")
    tiles = list_test_tiles()
    if tiles is None:
        st.error(f"File {FERTY / 'splits.json'} non trovato: lancia prima la preparazione dei dati "
                 "(src/datasets/slice_dataset.py).")
        return

    with st.sidebar:
        st.header("Object Detection")
        mode = st.radio("Modalità", ["Un modello", "Confronto di tutti i modelli"])
        if mode == "Un modello":
            model = st.selectbox("Architettura", list(MODELS), format_func=MODELS.get)
            regime = st.selectbox("Scenario di training", list(REGIMES), format_func=REGIMES.get)
        conf = st.slider("Soglia di confidenza", 0.05, 0.95, 0.25, 0.05,
                         help="Le detection con score inferiore non vengono mostrate")
        show_gt = st.checkbox("Mostra le annotazioni reali", value=True)
        st.markdown("---")
        st.subheader("Pesi disponibili")
        for m, mname in MODELS.items():
            st.caption(f"{mname}: " + " · ".join(
                f"{'✅' if find_weights(m, r) else '❌'} {r[-1].upper()}" for r in REGIMES))

    c1, c2 = st.columns([1, 3])
    with c1:
        sample = st.selectbox("Campione di test", DET_TEST_SAMPLES)
        names = tiles.get(sample, [])
        if not names:
            st.warning(f"Nessuna tile di {sample} in {FERTY / 'images'}")
            return
        key = f"tile_{sample}"
        if key not in st.session_state or st.session_state[key] not in names:
            st.session_state[key] = names[0]
        if st.button("Tile casuale"):
            st.session_state[key] = random.choice(names)
        tile = st.selectbox(f"Tile ({len(names)} disponibili)", names, key=key)
    image = Image.open(FERTY / "images" / tile).convert("RGB")
    W, H = image.size
    gt = load_gt(tile, W, H)
    with c2:
        st.markdown(f"**{tile}** · {W}×{H} px · **{len(gt)}** teste annotate  \n"
                    f"<span style='color:rgb{GT_COLOR}'>■</span> annotazione reale &nbsp; "
                    f"<span style='color:rgb{PRED_COLOR}'>■</span> predizione del modello",
                    unsafe_allow_html=True)

    def run_one(m, r):
        w = find_weights(m, r)
        if w is None:
            return None
        net, device = load_det_model(m, str(w))
        return predict(m, net, device, image, conf)

    if mode == "Un modello":
        w = find_weights(model, regime)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Immagine originale**" + (" con annotazioni" if show_gt else ""))
            st.image(draw_boxes(image, gt if show_gt else None), width="stretch")
        with col2:
            st.markdown(f"**{MODELS[model]} · {REGIMES[regime]}**")
            if w is None:
                st.error(f"Pesi non trovati per {model}_{regime} (attesi in runs/train/{model}_{regime}/weights/best.pt)")
                return
            with st.spinner(f"Inferenza con {MODELS[model]}..."):
                try:
                    pred, scores = run_one(model, regime)
                except Exception as e:
                    st.error(f"Errore durante l'inferenza: {e}")
                    return
            st.image(draw_boxes(image, gt if show_gt else None, pred), width="stretch")
        tp, p, r = tile_metrics(pred, scores, gt)
        k = st.columns(5)
        k[0].metric("Teste annotate", len(gt))
        k[1].metric("Teste rilevate", len(pred))
        k[2].metric("Corrette (IoU ≥ 0.5)", tp)
        k[3].metric("Precision (tile)", f"{p:.2f}")
        k[4].metric("Recall (tile)", f"{r:.2f}")
    else:
        st.image(draw_boxes(image, gt if show_gt else None), caption="Tile con le annotazioni reali", width=420)
        summary = []
        for m, mname in MODELS.items():
            cols = st.columns(2)
            for c, (r, rname) in zip(cols, REGIMES.items()):
                with c:
                    st.markdown(f"**{mname} · {rname.split(' (')[0]}**")
                    with st.spinner(f"{mname} {rname.split(' (')[0]}..."):
                        try:
                            res = run_one(m, r)
                        except Exception as e:
                            st.error(f"Errore: {e}")
                            continue
                    if res is None:
                        st.warning("Pesi non trovati")
                        continue
                    pred, scores = res
                    tp, p, rc = tile_metrics(pred, scores, gt)
                    st.image(draw_boxes(image, gt if show_gt else None, pred), width="stretch")
                    st.caption(f"rilevate {len(pred)} · corrette {tp}/{len(gt)} · P {p:.2f} · R {rc:.2f}")
                    summary.append({"Modello": mname, "Scenario": rname.split(" (")[0], "Rilevate": len(pred),
                                    "Corrette": tp, "Precision": round(p, 2), "Recall": round(rc, 2)})
        if summary:
            st.markdown("**Riepilogo sulla tile**")
            st.dataframe(summary, hide_index=True, width="stretch")

    st.caption("Precision e Recall sono calcolate solo su questa tile, alla soglia scelta: le metriche ufficiali sono "
               "quelle sull'intero test set.")
    off = official_test_metrics()
    with st.expander("Metriche ufficiali sull'intero test set (src/test.py)", expanded=False):
        if off:
            st.dataframe(off, hide_index=True, width="stretch")
        else:
            st.info("Nessun file runs/test/<modello>_<regime>/test_results.csv: lancia src/test.py.")


# TRACKING
def tracking_module():
    import src.tracking as trk
    return trk


@st.cache_data(show_spinner=False)
def list_test_videos():
    if not VIDEOS_DIR.exists():
        return None
    trk = tracking_module()
    vids = trk.discover_videos(VIDEOS_DIR)
    out = defaultdict(list)
    for (s, v) in sorted(vids):
        if s in TRACK_TEST_SAMPLES:
            out[s].append(v)
    return dict(out)


def tracker_params():
    f = TRACK_OUT / "tracker_params.json"
    if f.exists():
        with open(f, encoding="utf-8") as fh:
            return json.load(fh)["parametri"], True
    return dict(THESIS_PARAMS), False


def video_paths(sample, vid):
    avi = VIDEOS_DIR / sample / f"{sample}_{vid}.avi"
    js = avi.with_name(avi.stem + ".camera.json")
    return avi, (js if js.exists() else None)


def run_detection(avi, cache):
    # Detection YOLOv8 Scenario A su tutti i frame , salvata nella cache.
    import cv2
    trk = tracking_module()
    w = find_weights(TRACK_MODEL, TRACK_REGIME)
    if w is None:
        raise FileNotFoundError(f"Pesi non trovati: runs/train/{TRACK_MODEL}_{TRACK_REGIME}/weights/best.pt")
    detect = trk.load_detector(TRACK_MODEL, w, get_device(), TRACK_ARGS.det_conf)
    cap = cv2.VideoCapture(str(avi))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.release()
    bar = st.progress(0.0, text="Detection sui frame...")
    frame_dets, W, H = [], 0, 0
    for k, frame in trk.read_video_frames(avi):
        H, W = frame.shape[:2]
        grid = trk.tile_grid(W, H)
        dets = detect([frame[y:y + trk.TILE_SIZE, x:x + trk.TILE_SIZE] for x, y in grid])
        frame_dets.append((k, *trk.merge_tiles(dets, grid, W, H, TRACK_ARGS.merge_iou, TRACK_ARGS.merge_ios,
                                               TRACK_ARGS.edge_margin)))
        bar.progress(min((k + 1) / n, 1.0), text=f"Detection: frame {k + 1}/{n}")
    bar.empty()
    cache.parent.mkdir(parents=True, exist_ok=True)
    trk.save_cache(cache, frame_dets, W, H)


@st.cache_data(show_spinner=False)
def track_video(sample, vid, cache_mtime, params_json):
    # tracking + CASA dalle detection in cache 
    trk = tracking_module()
    p = json.loads(params_json)
    avi, js = video_paths(sample, vid)
    dets, W, H = trk.load_cache(TRACK_OUT / "det_cache" / f"{sample}_{vid}.npz")
    cam = trk.read_camera(js)
    fps = cam.get("fps") or trk.DEFAULT_FPS
    um = cam.get("um_per_px") or trk.DEFAULT_UM_PER_PX
    times = trk.frame_times(len(dets), cam, fps)
    rows = trk.run_tracker(dets, times, p, um, fps)
    casa = trk.compute_casa(rows, times, um, TRACK_ARGS)
    stats = trk.track_stats(rows, dets, TRACK_ARGS.min_track_len, p["track_high"])
    return {"rows": rows, "casa": casa, "stats": stats, "times": times, "W": W, "H": H, "um": um, "fps": fps,
            "n_frames": len(dets)}


@st.cache_data(show_spinner=False)
def trajectory_png(sample, vid, cache_mtime, params_json):
    trk = tracking_module()
    res = track_video(sample, vid, cache_mtime, params_json)
    avi, _ = video_paths(sample, vid)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "traj.png"
        trk.plot_trajectories(res["rows"], res["casa"], trk.first_frame_rgb(avi),
                              f"{sample}_{vid} [test] - YOLOv8 Scenario A - ByteTrack - "
                              f"{res['n_frames']} frame in {res['times'][-1]:.1f} s", out, TRACK_ARGS)
        return out.read_bytes()


def read_frame(avi, k):
    import cv2
    cap = cv2.VideoCapture(str(avi))
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, frame = cap.read()
    cap.release()
    return Image.fromarray(frame[:, :, ::-1]) if ok else None


def draw_tracks_on_frame(img, rows, casa, k, tail=15):
    cls = {c["track_id"]: c["classe"] for c in casa}
    hist = defaultdict(list)
    for f, tid, x1, y1, x2, y2, _ in rows:
        if k - tail <= f <= k:
            hist[tid].append((f, (x1 + x2) / 2, (y1 + y2) / 2, (x1, y1, x2, y2)))
    d = ImageDraw.Draw(img)
    for tid, pts in hist.items():
        pts.sort()
        col = hex_rgb(CLASS_COLORS.get(cls.get(tid), SHORT_COLOR))
        if len(pts) > 1:
            d.line([(x, y) for _, x, y, _ in pts], fill=col, width=2)
        f, _, _, box = pts[-1]
        if f == k:
            d.rectangle(list(box), outline=col, width=2)
            d.text((box[2] + 2, box[1] - 10), str(tid), fill=col)
    return img


def mot_text(rows, casa):
    keep = {c["track_id"] for c in casa}
    return "".join(f"{k + 1},{tid},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{sc:.3f},-1,-1,-1\n"
                   for k, tid, x1, y1, x2, y2, sc in sorted(rows) if tid in keep)


def to_csv(dict_rows):
    if not dict_rows:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(dict_rows[0].keys()))
    w.writeheader()
    w.writerows(dict_rows)
    return buf.getvalue()


def tracking_page():
    st.subheader("Tracking e analisi CASA sui video di test")
    st.caption("Detector: YOLOv8, Scenario A · Tracker: ByteTrack con associazione per distanza dei centri e tempo "
               "reale dei frame · Classi di motilità OMS dalle tracce di almeno 30 punti.")
    videos = list_test_videos()
    if videos is None:
        st.error(f"Cartella dei video non trovata: {VIDEOS_DIR}")
        return
    if not videos:
        st.warning("Nessun video dei campioni di test in data/spai_video.")
        return
    p, from_file = tracker_params()

    with st.sidebar:
        st.header("Tracking")
        st.markdown("**Modello:** YOLOv8 · Scenario A")
        w = find_weights(TRACK_MODEL, TRACK_REGIME)
        st.caption(("✅ pesi trovati" if w else "❌ pesi non trovati") + " · "
                   + ("parametri da tracker_params.json" if from_file else "parametri della tesi (Tabella 5.5)"))
        st.markdown(f"v_max **{p['max_speed']} µm/s** · τ_high **{p['track_high']}** · "
                    f"nuova traccia **{p['new_track']}** · track buffer **{p['track_buffer']}**")

    c1, c2 = st.columns(2)
    sample = c1.selectbox("Campione di test", [s for s in TRACK_TEST_SAMPLES if s in videos],
                          help="Campioni di test del tracking (SEM-0012 e SEM-0013 esclusi: senza spermatozoi)")
    vid = c2.selectbox("Parte del video (circa 5 s)", videos[sample])
    avi, _ = video_paths(sample, vid)
    cache = TRACK_OUT / "det_cache" / f"{sample}_{vid}.npz"

    if not cache.exists():
        st.info("Le detection di questo video non sono ancora in cache (src/tracking.py --stage detect). "
                "Puoi calcolarle ora: con le 6 tile per frame servono alcuni minuti.")
        if st.button("Esegui detection e tracking"):
            try:
                run_detection(avi, cache)
            except Exception as e:
                st.error(f"Errore durante la detection: {e}")
                return
            st.rerun()
        return

    params_json = json.dumps(p, sort_keys=True)
    mtime = cache.stat().st_mtime
    with st.spinner("Tracking e calcolo CASA..."):
        res = track_video(sample, vid, mtime, params_json)
    casa, rows, stats = res["casa"], res["rows"], res["stats"]

    n = len(casa)
    cnt = {c: sum(r["classe"] == c for r in casa) for c in CLASS_COLORS}
    pct = lambda x: f"{100 * x / n:.1f}%" if n else "-"
    med = lambda key: f"{np.median([r[key] for r in casa]):.1f}" if n else "-"
    k = st.columns(6)
    k[0].metric("Tracce valide", n)
    k[1].metric("Progressive", pct(cnt["progressivo rapido"] + cnt["progressivo lento"]))
    k[2].metric("Rapide", pct(cnt["progressivo rapido"]))
    k[3].metric("Lente", pct(cnt["progressivo lento"]))
    k[4].metric("Non progressive", pct(cnt["non progressivo"]))
    k[5].metric("Immobili", pct(cnt["immobile"]))
    k = st.columns(6)
    k[0].metric("VCL mediana", f"{med('VCL')} µm/s")
    k[1].metric("VAP mediana", f"{med('VAP')} µm/s")
    k[2].metric("VSL mediana", f"{med('VSL')} µm/s")
    k[3].metric("Frame", res["n_frames"])
    k[4].metric("Copertura", f"{stats['copertura']:.2f}")
    k[5].metric("Continuità", f"{stats['continuita']:.2f}")

    tab1, tab2, tab3 = st.tabs(["Traiettorie", "Frame per frame", "Parametri CASA"])
    with tab1:
        with st.spinner("Disegno delle traiettorie..."):
            st.image(trajectory_png(sample, vid, mtime, params_json), width="stretch")
        gif = TRACK_OUT / "trajectories" / f"{sample}_{vid}.gif"
        if gif.exists():
            with st.expander("Animazione (GIF prodotta da src/tracking.py --gif)"):
                st.image(str(gif))
    with tab2:
        f = st.slider("Frame", 0, res["n_frames"] - 1, 0)
        img = read_frame(avi, f)
        if img is None:
            st.warning("Impossibile leggere il frame dal video.")
        else:
            st.image(draw_tracks_on_frame(img, rows, casa, f), width="stretch",
                     caption=f"Frame {f} · t = {res['times'][f]:.2f} s · box e scia degli ultimi 15 frame, "
                             f"colore = classe di motilità (grigio = traccia sotto i 30 punti)")
    with tab3:
        if casa:
            st.dataframe(casa, hide_index=True, width="stretch")
        else:
            st.info("Nessuna traccia di almeno 30 punti in questo video.")
        d1, d2 = st.columns(2)
        d1.download_button("Pseudo-etichette (formato MOT)", mot_text(rows, casa),
                           file_name=f"{sample}_{vid}_mot.txt", disabled=not casa)
        d2.download_button("Parametri CASA (CSV)", to_csv(casa), file_name=f"{sample}_{vid}_casa.csv",
                           disabled=not casa)

    summary = TRACK_OUT / "casa_per_campione.csv"
    with st.expander("Riepilogo su tutti i campioni di test (src/tracking.py --stage casa)"):
        if summary.exists():
            rows_s = [r for r in read_csv_dicts(summary) if r.get("insieme") == "test"]
            cols = ["campione", "video", "tracce", "progressivi %", "progressivo rapido %", "progressivo lento %",
                    "non progressivo %", "immobile %", "VCL mediana", "VAP mediana", "VSL mediana"]
            st.dataframe([{c: r.get(c) for c in cols} for r in rows_s], hide_index=True, width="stretch")
        else:
            st.info("File casa_per_campione.csv non trovato: lancia src/tracking.py --stage casa.")


# Pagina
st.title("Rilevamento e tracking di spermatozoi")
st.markdown("Demo della tesi *Sviluppo di un Sistema di Pseudo-Labelling per il Multi-Object Tracking "
            "Automatizzato di Spermatozoi*: tutte le prove sono fatte **solo sui campioni di test**.")

page = st.sidebar.radio("Sezione", ["Object Detection", "Tracking"])
st.sidebar.markdown("---")
if page == "Object Detection":
    detection_page()
else:
    tracking_page()