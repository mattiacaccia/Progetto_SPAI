import numpy as np

IOU_THRS = np.linspace(0.5, 0.95, 10)


def box_iou_np(a, b):

    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def _match_image(scores, ious, thr):
    order = np.argsort(-scores, kind="stable")
    n_pred, n_gt = ious.shape
    is_tp = np.zeros(n_pred, dtype=bool)
    if n_gt == 0:
        return order, is_tp
    gt_taken = np.zeros(n_gt, dtype=bool)
    for rank, p in enumerate(order):
        cand = np.where(gt_taken, -1.0, ious[p])
        g = int(cand.argmax())
        if cand[g] >= thr:
            gt_taken[g] = True
            is_tp[rank] = True
    return order, is_tp


def _ap_from_pr(tp_cum, fp_cum, n_gt):
    if n_gt == 0 or len(tp_cum) == 0:
        return 0.0
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    precision = np.flip(np.maximum.accumulate(np.flip(precision)))  # envelope monotona
    x = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, x, side="left")
    q = np.zeros(101)
    valid = idx < len(precision)
    q[valid] = precision[idx[valid]]
    return float(q.mean())


def compute_detection_metrics(preds, gts, min_score=0.001, fixed_conf=0.5):

    n_gt_total = int(sum(len(g) for g in gts))
    per_thr_scores = [[] for _ in IOU_THRS]
    per_thr_tp = [[] for _ in IOU_THRS]

    for (pb, ps), gb in zip(preds, gts):
        pb = np.asarray(pb, dtype=np.float64).reshape(-1, 4)
        ps = np.asarray(ps, dtype=np.float64).reshape(-1)
        gb = np.asarray(gb, dtype=np.float64).reshape(-1, 4)
        keep = ps >= min_score
        pb, ps = pb[keep], ps[keep]
        if len(ps) == 0:
            continue
        ious = box_iou_np(pb, gb)
        for t, thr in enumerate(IOU_THRS):
            order, is_tp = _match_image(ps, ious, thr)
            per_thr_scores[t].append(ps[order])
            per_thr_tp[t].append(is_tp)

    aps = []
    curves = None
    for t in range(len(IOU_THRS)):
        if per_thr_scores[t]:
            s = np.concatenate(per_thr_scores[t])
            tp = np.concatenate(per_thr_tp[t])
        else:
            s, tp = np.zeros(0), np.zeros(0, dtype=bool)
        o = np.argsort(-s, kind="stable")
        s, tp = s[o], tp[o]
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(~tp)
        aps.append(_ap_from_pr(tp_cum, fp_cum, n_gt_total))
        if t == 0:  # IoU 0.5: serve per P/R/F1
            curves = (s, tp_cum, fp_cum)

    s, tp_cum, fp_cum = curves
    out = {"mAP50": aps[0], "mAP50-95": float(np.mean(aps))}

    def prf(tp, fp):
        p = tp / max(tp + fp, 1e-12)
        r = tp / max(n_gt_total, 1e-12)
        f = 2 * p * r / max(p + r, 1e-12)
        return float(p), float(r), float(f)

    # punto operativo a soglia fissa
    k = int((s >= fixed_conf).sum())
    out["fixed"] = prf(tp_cum[k - 1], fp_cum[k - 1]) if k > 0 else (0.0, 0.0, 0.0)
    out["fixed_conf"] = fixed_conf

    # punto operativo che massimizza F1 (come Ultralytics)
    if len(s) > 0:
        p = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        r = tp_cum / max(n_gt_total, 1e-12)
        f = 2 * p * r / np.maximum(p + r, 1e-12)
        i = int(f.argmax())
        out["best_f1"] = (float(p[i]), float(r[i]), float(f[i]))
        out["best_conf"] = float(s[i])
    else:
        out["best_f1"] = (0.0, 0.0, 0.0)
        out["best_conf"] = 0.0
    return out