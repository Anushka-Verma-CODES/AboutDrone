"""Drone-model identification for tracked drones.

The classifier (YOLO11s-cls, trained on VisioDECT crops) is run on a few of
each track's most confident boxes, cropped the same way as the training crops
(square, 1.3x the box). The class probabilities are averaged over those
crops, which is steadier than trusting a single frame.

It only knows the six VisioDECT models, so a drone of another type is still
mapped to one of them. Low average confidence is reported as "uncertain".
"""
import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.config import MODELS
from src.tracking.track import iter_frames

DEFAULT_CLS_WEIGHTS = MODELS / "classifier" / "best.pt"
CROPS_PER_TRACK = 12
MIN_CROP_PX = 8
UNCERTAIN_BELOW = 0.5


def square_crop(frame, x1, y1, x2, y2, pad=1.3):
    h, w = frame.shape[:2]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = max(x2 - x1, y2 - y1) * pad
    a, b = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    c, d = int(min(w, cx + s / 2)), int(min(h, cy + s / 2))
    if c - a < MIN_CROP_PX or d - b < MIN_CROP_PX:
        return None
    return frame[b:d, a:c]


def pick_boxes(tracks, k=CROPS_PER_TRACK):
    """Per track: the k most confident boxes, at most one per 5 frames so the
    crops are not all near-copies of the same moment."""
    picked = []
    for _, g in tracks.sort_values("conf", ascending=False).groupby("track_id", sort=False):
        used = []
        for r in g.itertuples():
            if all(abs(r.frame - f) >= 5 for f in used):
                used.append(r.frame)
                picked.append(r)
                if len(used) == k:
                    break
    return picked


def classify_tracks(source, tracks, weights=DEFAULT_CLS_WEIGHTS, device=0, k=CROPS_PER_TRACK):
    """Returns one row per track: track_id, drone_model, model_conf, runner_up, crops."""
    cols = ["track_id", "drone_model", "model_conf", "runner_up", "crops"]
    if not len(tracks):
        return pd.DataFrame(columns=cols)
    model = YOLO(str(weights))
    names = model.names
    wanted = {}
    for r in pick_boxes(tracks, k):
        wanted.setdefault(int(r.frame), []).append(r)
    last = max(wanted)

    crops, owners = [], []
    for i, frame in enumerate(iter_frames(source)):
        for r in wanted.get(i, ()):
            c = square_crop(frame, r.x1, r.y1, r.x2, r.y2)
            if c is not None:
                crops.append(c)
                owners.append(r.track_id)
        if i >= last:
            break

    probs = {}
    for s in range(0, len(crops), 32):
        for res, tid in zip(model.predict(crops[s:s + 32], device=device, verbose=False),
                            owners[s:s + 32]):
            probs.setdefault(tid, []).append(res.probs.data.cpu().numpy())

    rows = []
    for tid in sorted(tracks.track_id.unique()):
        if tid not in probs:
            rows.append({"track_id": tid, "drone_model": "box too small", "model_conf": np.nan,
                         "runner_up": "", "crops": 0})
            continue
        p = np.mean(probs[tid], axis=0)
        first, second = np.argsort(-p)[:2]
        label = names[int(first)]
        rows.append({"track_id": tid,
                     "drone_model": label if p[first] >= UNCERTAIN_BELOW else f"uncertain ({label}?)",
                     "model_conf": float(p[first]),
                     "runner_up": f"{names[int(second)]} ({p[second]:.0%})",
                     "crops": len(probs[tid])})
    return pd.DataFrame(rows, columns=cols)
