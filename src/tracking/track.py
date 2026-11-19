"""Detection + ByteTrack over a video file or a folder of frames."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.config import ROOT

TRACKER_CFG = str(ROOT / "configs" / "bytetrack_drone.yaml")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_frames(source):
    """Yield BGR frames from a video path or a directory of images."""
    source = Path(source)
    if source.is_dir():
        for p in sorted(f for f in source.iterdir() if f.suffix.lower() in IMG_EXT):
            yield cv2.imread(str(p))
    else:
        cap = cv2.VideoCapture(str(source))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
        cap.release()


def video_fps(source, default=25.0):
    source = Path(source)
    if source.is_dir():
        return default
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 1 else default


def track(source, weights, conf=0.1, imgsz=1280, device=0, max_frames=None, on_frame=None):
    """Run the detector with ByteTrack. Returns a DataFrame with one row per
    (frame, track): frame, track_id, x1, y1, x2, y2, conf."""
    model = YOLO(str(weights))
    rows = []
    for i, frame in enumerate(iter_frames(source)):
        if max_frames and i >= max_frames:
            break
        res = model.track(frame, persist=True, tracker=TRACKER_CFG, conf=conf,
                          imgsz=imgsz, device=device, verbose=False)[0]
        b = res.boxes
        if b is not None and b.id is not None:
            ids = b.id.int().cpu().numpy()
            xyxy = b.xyxy.cpu().numpy()
            cf = b.conf.cpu().numpy()
            for tid, box, c in zip(ids, xyxy, cf):
                rows.append((i, int(tid), *box.tolist(), float(c)))
        if on_frame:
            on_frame(i, frame, res)
    return pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "conf"])


def detect_only(source, weights, conf=0.1, imgsz=1280, device=0):
    """Highest-confidence box per frame (used for single-object evaluation)."""
    model = YOLO(str(weights))
    rows = []
    for i, frame in enumerate(iter_frames(source)):
        r = model.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        if len(r.boxes):
            k = int(r.boxes.conf.argmax())
            rows.append((i, *r.boxes.xyxy[k].cpu().numpy().tolist(), float(r.boxes.conf[k])))
    return pd.DataFrame(rows, columns=["frame", "x1", "y1", "x2", "y2", "conf"])


def tracks_to_segments(df, fps, min_len=15):
    """Convert tracker output into trajectory dicts for the feature code."""
    out = {}
    for tid, g in df.groupby("track_id"):
        if len(g) < min_len:
            continue
        g = g.sort_values("frame")
        w = (g.x2 - g.x1).to_numpy()
        h = (g.y2 - g.y1).to_numpy()
        out[int(tid)] = {
            "frame": g.frame.to_numpy(),
            "t": g.frame.to_numpy() / fps,
            "u": ((g.x1 + g.x2) / 2).to_numpy(),
            "v": -((g.y1 + g.y2) / 2).to_numpy(),    # image y points down
            "scale": np.sqrt(np.clip(w * h, 1, None)),
        }
    return out
