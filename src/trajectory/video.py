"""Real video trajectories (DUT Anti-UAV tracking videos) for the anomaly models.

The 20 videos are split by video into train / val / test (12 / 4 / 4), so no
part of a test video is ever seen in training.

- train: windows of every track, assumed normal (ordinary flights). They
  teach the unsupervised models what real, detector-produced tracks look like
  (residual jitter, camera motion, size changes) - something neither the
  synthetic set nor the telemetry has.
- val / test: the real tracks as normal windows, plus labelled anomalies made
  by injecting the synthetic anomaly types into copies of the same real
  tracks. The injection adds the anomaly's displacement on top of the real
  motion, so detector noise and camera motion are kept.

Input is the raw tracker output saved by src.tracking.evaluate
(reports/dut_tracks_videoNN.csv), so this step does not rerun the detector.

    python -m src.trajectory.video
"""
import numpy as np
import pandas as pd

from src.config import REPORTS, SEED, STRIDE, TRAJ_DATA, TRAJ_HZ
from src.tracking.camera import load_cam
from src.tracking.track import tracks_to_segments
from src.trajectory.features import merge, resample, windows_from_segment
from src.trajectory.synth import ANOMALY_TYPES, inject

DUT_FPS = 25.0
N_VAL, N_TEST = 4, 4
MIN_INJECT_LEN = 60      # samples (6 s): room for an injected interval plus context


def video_split(videos, seed=SEED):
    order = list(np.random.default_rng(seed).permutation(sorted(videos)))
    return {"test": order[:N_TEST], "val": order[N_TEST:N_TEST + N_VAL], "train": order[N_TEST + N_VAL:]}


def segments(df, smooth=True, cam=None):
    """Resampled 10 Hz segments [(track_id, seg)] of one video's tracks."""
    out = []
    for tid, tr in tracks_to_segments(df, DUT_FPS, smooth=smooth, cam=cam).items():
        for s in resample(tr["t"], tr["u"], tr["v"], tr["scale"]):
            out.append((tid, s))
    return out


def _windows(seg, group, cls, stride):
    w = windows_from_segment(seg, stride=stride)
    if w is None:
        return None
    w["group"] = np.array([group] * len(w["y"]))
    w["cls"] = np.array([cls] * len(w["y"]))
    return w


def inject_real(seg, kind, rng):
    """Copy of a real segment with one synthetic anomaly added on top."""
    s = float(np.median(seg["scale"])) or 1.0
    dt = 1.0 / TRAJ_HZ
    u, v = seg["u"] / s, seg["v"] / s
    vx0, vy0 = np.gradient(u, dt), np.gradient(v, dt)
    vx, vy = vx0.copy(), vy0.copy()
    off = np.zeros((len(u), 2))
    lab = inject(rng, vx, vy, off, kind)
    du = np.cumsum(vx - vx0) * dt + off[:, 0]
    dv = np.cumsum(vy - vy0) * dt + off[:, 1]
    out = dict(seg)
    out["u"], out["v"] = (u + du) * s, (v + dv) * s
    out["label"] = lab
    return out


def build(video_names, smooth=True, with_injected=False, stride=STRIDE, seed=SEED, camera=True):
    rng = np.random.default_rng(seed)
    parts = []
    for name in video_names:
        df = pd.read_csv(REPORTS / f"dut_tracks_{name}.csv")
        cam = load_cam(name) if camera else None
        for tid, seg in segments(df, smooth, cam):
            parts.append(_windows(seg, f"{name}_{tid}", "real_video", stride))
            if not with_injected or len(seg["t"]) < MIN_INJECT_LEN:
                continue
            for kind in ANOMALY_TYPES:
                w = _windows(inject_real(seg, kind, rng), f"{name}_{tid}_{kind}", kind, STRIDE)
                if w is not None:   # keep only the windows that contain the anomaly
                    keep = w["y"] == 1
                    parts.append({k: x[keep] for k, x in w.items()})
    return merge(parts)


def main():
    videos = sorted(p.stem.replace("dut_tracks_", "") for p in REPORTS.glob("dut_tracks_video*.csv"))
    if not videos:
        raise SystemExit("no tracker output found, run src.tracking.evaluate first")
    split = video_split(videos)
    print({k: v for k, v in split.items()})

    # stride 1 for training: more (overlapping) real windows from few videos
    data = {
        "train": build(split["train"], stride=1),
        "val": build(split["val"], with_injected=True),
        "test": build(split["test"], with_injected=True),
    }
    for k, d in data.items():
        np.savez_compressed(TRAJ_DATA / f"video_{k}.npz", **d)
        print(f"video_{k}: {len(d['y'])} windows, {int(d['y'].sum())} anomalous")

    # ablations on the same test videos: no Kalman smoothing / no camera compensation
    raw = build(split["test"], smooth=False, with_injected=True)
    np.savez_compressed(TRAJ_DATA / "video_test_unsmoothed.npz", **raw)
    nocam = build(split["test"], with_injected=True, camera=False)
    np.savez_compressed(TRAJ_DATA / "video_test_no_camera_comp.npz", **nocam)
    # all videos, normal windows only (used by older figures / sanity checks)
    np.savez_compressed(TRAJ_DATA / "dut_video_tracks.npz", **build(videos))
    (TRAJ_DATA / "video_split.json").write_text(pd.Series(split).to_json(indent=2))


if __name__ == "__main__":
    main()
