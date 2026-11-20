"""Video -> detection -> tracking -> trajectory features -> anomaly verdict.

    python -m src.pipeline --source path/to/video.mp4 --out outputs/demo
"""
import argparse
import json
import pickle
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch

from src.anomaly.models import LSTMAEDetector
from src.config import MODELS, OUTPUTS, TRAJ_HZ, WINDOW
from src.tracking.link import MIN_TRACK_CONF, filter_tracks, link_tracks
from src.tracking.track import DEFAULT_TRACKER, iter_frames, track, tracks_to_segments, video_fps
from src.trajectory.features import SUMMARY_NAMES, resample, windows_from_segment

DEFAULT_WEIGHTS = MODELS / "detector" / "best.pt"
ANOM = MODELS / "anomaly"


class AnomalyScorer:
    """Loads the trained anomaly models. `source` picks the operating point
    (threshold) tuned for that kind of input: "video" (default, what the
    pipeline sees), "telemetry", "synthetic", or "global"."""

    def __init__(self, model=None, source="video"):
        with open(ANOM / "calibration.pkl", "rb") as f:
            self.cal = pickle.load(f)
        self.model = model or self.cal["selected"]
        self.iforest = joblib.load(ANOM / "isolation_forest.joblib")
        self.ocsvm = joblib.load(ANOM / "one_class_svm.joblib")
        self.gbm = joblib.load(ANOM / "supervised_gbm.joblib")
        st = torch.load(ANOM / "lstm_autoencoder.pt", map_location="cpu", weights_only=False)
        self.lstm = LSTMAEDetector.from_state(st, device="cpu")   # tiny model, keep GPU for YOLO
        self.use(self.model, source)

    def threshold_for(self, model, source):
        return self.cal.get("source_thresholds", {}).get(f"{source}/{model}", self.cal["thresholds"][model])

    def use(self, model=None, source=None):
        """Switch model and/or operating point."""
        self.model = model or self.model
        self.source = source or getattr(self, "source", "video")
        self.threshold = self.threshold_for(self.model, self.source)
        return self

    def typical_normal(self):
        """Median score of normal validation windows (colour scale baseline).
        The ensemble score is already a percentile, so its median is 0.5."""
        ref = self.cal["calib_ref"].get(self.model)
        return 0.5 if ref is None else float(np.percentile(ref, 50))

    def _pct(self, name, s):
        ref = self.cal["calib_ref"][name]
        return np.searchsorted(ref, s) / len(ref)

    def raw(self, w, name):
        if name == "isolation_forest":
            return self.iforest.score(w["summary"])
        if name == "one_class_svm":
            return self.ocsvm.score(w["summary"])
        if name == "supervised_gbm":
            return self.gbm.score(w["summary"])
        if name == "lstm_autoencoder":
            return self.lstm.score(w["seq"])
        if name == "ensemble":
            return 0.5 * (self._pct("isolation_forest", self.raw(w, "isolation_forest")) +
                          self._pct("lstm_autoencoder", self.raw(w, "lstm_autoencoder")))
        raise ValueError(name)

    def score(self, w):
        s = self.raw(w, self.model)
        return s, s >= self.threshold

    def reasons(self, summary, top=2):
        z = (summary - self.cal["normal_median"]) / self.cal["normal_iqr"]
        out = []
        for row in z:
            idx = np.argsort(-row)[:top]
            out.append(", ".join(f"{SUMMARY_NAMES[i]} (+{row[i]:.1f} IQR)" for i in idx if row[i] > 1.5))
        return out


def score_segments(segs_by_id, scorer):
    """Score every track. Returns (per-window DataFrame, per-track DataFrame)."""
    win_rows, track_rows = [], []
    for tid, tr in segs_by_id.items():
        parts = resample(tr["t"], tr["u"], tr["v"], tr["scale"], max_gap=1.0)
        n_win, n_anom, best = 0, 0, -np.inf
        for seg in parts:
            w = windows_from_segment(seg)
            if w is None:
                continue
            s, flag = scorer.score(w)
            why = scorer.reasons(w["summary"])
            for t0, si, fi, wy in zip(w["t0"], s, flag, why):
                win_rows.append({"track_id": tid, "t_start": float(t0),
                                 "t_end": float(t0) + WINDOW / TRAJ_HZ,
                                 "score": float(si), "anomalous": bool(fi), "reason": wy})
            n_win += len(s)
            n_anom += int(flag.sum())
            best = max(best, float(s.max()))
        track_rows.append({
            "track_id": tid,
            "duration_s": float(tr["t"][-1] - tr["t"][0]),
            "windows": n_win,
            "anomalous_windows": n_anom,
            "max_score": best if n_win else np.nan,
            "verdict": ("too short" if n_win == 0 else
                        "ANOMALOUS" if n_anom >= max(2, 0.1 * n_win) else "normal"),
        })
    return pd.DataFrame(win_rows), pd.DataFrame(track_rows)


def frame_scores(windows, fps):
    """Per (track, frame) anomaly score = max over windows covering that frame."""
    lut = {}
    for r in windows.itertuples():
        for f in range(int(r.t_start * fps), int(r.t_end * fps) + 1):
            k = (r.track_id, f)
            lut[k] = max(lut.get(k, -np.inf), r.score)
    return lut


def color_for(score, thr, lo):
    """Green -> yellow -> red as the score crosses the threshold."""
    if not np.isfinite(score):
        return (200, 200, 200)
    x = np.clip((score - lo) / max(thr - lo, 1e-9), 0, 1.5) / 1.5
    return (0, int(255 * (1 - x)), int(255 * min(1, 2 * x)))


def render(source, tracks_df, windows, scorer, fps, out_path, trail=60):
    lut = frame_scores(windows, fps) if len(windows) else {}
    lo = scorer.typical_normal()
    thr = scorer.threshold
    by_frame = {f: g for f, g in tracks_df.groupby("frame")}
    history = {}
    writer = None
    for i, frame in enumerate(iter_frames(source)):
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if i in by_frame:
            for r in by_frame[i].itertuples():
                c = (int((r.x1 + r.x2) / 2), int((r.y1 + r.y2) / 2))
                s = lut.get((r.track_id, i), np.nan)
                history.setdefault(r.track_id, []).append((c, s))
                col = color_for(s, thr, lo)
                cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), col, 2)
                tag = f"ID {r.track_id}" + ("" if np.isnan(s) else f"  {'ANOM' if s >= thr else 'ok'}")
                cv2.putText(frame, tag, (int(r.x1), int(r.y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        for tid, pts in history.items():
            pts = pts[-trail:]
            for (p0, _), (p1, s1) in zip(pts[:-1], pts[1:]):
                cv2.line(frame, p0, p1, color_for(s1, thr, lo), 2)
        writer.write(frame)
    if writer:
        writer.release()


def run(source, out_dir, weights=DEFAULT_WEIGHTS, conf=0.1, imgsz=1280, model=None,
        max_frames=None, render_video=True, device=0, threshold_source="video",
        camera_comp=True, tracker=DEFAULT_TRACKER, link=True, min_track_conf=MIN_TRACK_CONF):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = video_fps(source)
    tr = track(source, weights, conf=conf, imgsz=imgsz, max_frames=max_frames, device=device,
               tracker=tracker)
    if link:
        tr = link_tracks(tr, fps)[0]
    if min_track_conf:
        tr = filter_tracks(tr, min_track_conf)
    tr.to_csv(out_dir / "tracks.csv", index=False)
    cam = None
    if camera_comp:
        from src.tracking.camera import estimate_motion
        cam, _ = estimate_motion(source, max_frames=max_frames)
    segs = tracks_to_segments(tr, fps, cam=cam)
    scorer = AnomalyScorer(model, threshold_source)
    win, per_track = score_segments(segs, scorer)
    win.to_csv(out_dir / "windows.csv", index=False)
    per_track.to_csv(out_dir / "track_summary.csv", index=False)
    if render_video:
        render(source, tr, win, scorer, fps, out_dir / "annotated.mp4")
    (out_dir / "meta.json").write_text(json.dumps(
        {"source": str(source), "fps": fps, "model": scorer.model,
         "threshold": scorer.threshold, "threshold_source": scorer.source,
         "camera_compensation": camera_comp, "tracker": tracker, "linking": link, "min_track_conf": min_track_conf, "n_tracks": int(tr.track_id.nunique()) if len(tr) else 0},
        indent=2, default=float))
    return tr, win, per_track


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default=str(OUTPUTS / "run"))
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--model", default=None)
    ap.add_argument("--threshold-source", default="video",
                    choices=["video", "telemetry", "synthetic", "global"],
                    help="which validation set the decision threshold was tuned on")
    ap.add_argument("--tracker", default=DEFAULT_TRACKER, choices=["bytetrack", "botsort"])
    ap.add_argument("--no-linking", action="store_true", help="skip offline tracklet linking")
    ap.add_argument("--no-camera-comp", action="store_true", help="skip camera motion compensation")
    ap.add_argument("--min-track-conf", type=float, default=MIN_TRACK_CONF,
                    help="drop tracks whose best detection is below this (0 = keep all)")
    a = ap.parse_args()
    _, _, summary = run(a.source, a.out, a.weights, a.conf, a.imgsz, a.model,
                        threshold_source=a.threshold_source, camera_comp=not a.no_camera_comp,
                        tracker=a.tracker, link=not a.no_linking, min_track_conf=a.min_track_conf)
    print(summary.to_string(index=False))
