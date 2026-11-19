"""Unit-free kinematic features shared by every trajectory source.

All three sources describe a trajectory as
    t      (N,)   seconds
    u, v   (N,)   horizontal / vertical position, v pointing UP
    scale  (N,)   size of the drone in the same unit as u, v

- video:      u, v in pixels (v flipped), scale = sqrt(bbox w*h)
- telemetry:  u, v in metres (projected), scale = nominal frame size (0.5 m)
- synthetic:  already in body lengths, scale = 1

Dividing positions by scale gives "body lengths", so a drone 20 px wide at
100 m and a 0.5 m frame in a log file end up in comparable units.
"""
import numpy as np
from scipy.signal import savgol_filter

from src.config import STRIDE, TRAJ_HZ, WINDOW

HOVER_SPEED = 0.3   # body lengths / s
CHANNELS = ["speed", "v_vert", "acc_tan", "acc_norm", "jerk"]

SUMMARY_NAMES = [
    "speed_mean", "speed_std", "speed_max", "speed_p90",
    "vvert_absmean", "vvert_absmax",
    "acctan_absmax", "accnorm_mean", "accnorm_max",
    "jerk_mean", "jerk_max",
    "turn_absmean", "turn_absmax",
    "straightness", "hover_ratio", "reversals",
]


def slog(x):
    """Sign-preserving log, tames the heavy tails of acc / jerk."""
    return np.sign(x) * np.log1p(np.abs(x))


def resample(t, u, v, scale, label=None, hz=TRAJ_HZ, max_gap=1.0):
    """Interpolate onto a uniform grid. Splits the track wherever there is a
    gap longer than max_gap seconds and returns a list of segments."""
    t = np.asarray(t, float)
    order = np.argsort(t)
    t, u, v, scale = t[order], np.asarray(u, float)[order], np.asarray(v, float)[order], np.asarray(scale, float)[order]
    if label is not None:
        label = np.asarray(label, float)[order]

    cut = np.where(np.diff(t) > max_gap)[0] + 1
    segs = []
    for idx in np.split(np.arange(len(t)), cut):
        if len(idx) < 4:
            continue
        ts = t[idx]
        grid = np.arange(ts[0], ts[-1], 1.0 / hz)
        if len(grid) < 4:
            continue
        seg = {
            "t": grid,
            "u": np.interp(grid, ts, u[idx]),
            "v": np.interp(grid, ts, v[idx]),
            "scale": np.interp(grid, ts, scale[idx]),
        }
        if label is not None:
            seg["label"] = np.interp(grid, ts, label[idx]) > 0.5
        segs.append(seg)
    return segs


def kinematics(seg, hz=TRAJ_HZ, smooth=True):
    """Per-step channels (N, 5) plus the turn rate used by summary stats."""
    s = np.median(seg["scale"])
    s = s if s > 0 else 1.0
    u, v = seg["u"] / s, seg["v"] / s
    n = len(u)
    if smooth and n >= 7:
        u = savgol_filter(u, 7, 2)
        v = savgol_filter(v, 7, 2)

    dt = 1.0 / hz
    vu, vv = np.gradient(u, dt), np.gradient(v, dt)
    au, av = np.gradient(vu, dt), np.gradient(vv, dt)
    ju, jv = np.gradient(au, dt), np.gradient(av, dt)

    speed = np.hypot(vu, vv)
    safe = np.maximum(speed, 1e-6)
    acc_tan = (au * vu + av * vv) / safe
    acc_norm = np.abs(au * vv - av * vu) / safe
    jerk = np.hypot(ju, jv)

    heading = np.unwrap(np.arctan2(vv, vu))
    turn = np.gradient(heading, dt)
    turn[speed < HOVER_SPEED] = 0.0   # heading is noise when hovering

    ch = np.stack([speed, vv, acc_tan, acc_norm, jerk], axis=1)
    return ch, turn, u, v


def window_summary(ch, turn, u, v):
    speed, vv, acc_tan, acc_norm, jerk = ch.T
    step = np.hypot(np.diff(u), np.diff(v))
    path = step.sum()
    straight = np.hypot(u[-1] - u[0], v[-1] - v[0]) / path if path > 1e-6 else 1.0

    moving = speed >= HOVER_SPEED
    h = np.unwrap(np.arctan2(np.gradient(v), np.gradient(u)))
    dh = np.diff(h)[moving[1:]]
    reversals = int(np.sum(np.abs(dh) > np.pi / 2))

    return np.array([
        speed.mean(), speed.std(), speed.max(), np.percentile(speed, 90),
        np.abs(vv).mean(), np.abs(vv).max(),
        np.abs(acc_tan).max(), acc_norm.mean(), acc_norm.max(),
        jerk.mean(), jerk.max(),
        np.abs(turn).mean(), np.abs(turn).max(),
        straight, (~moving).mean(), reversals,
    ], dtype=np.float32)


def windows_from_segment(seg, window=WINDOW, stride=STRIDE, min_anom_frac=0.2):
    """Cut a resampled segment into windows.

    Returns dict with
        seq      (W, window, 5)  slog-scaled per-step channels
        summary  (W, 16)         slog-scaled window statistics
        y        (W,)            1 if >= min_anom_frac of the window is labelled anomalous
        t0       (W,)            window start time
    """
    n = len(seg["t"])
    if n < window:
        return None
    ch, turn, u, v = kinematics(seg)
    seqs, sums, ys, t0 = [], [], [], []
    lab = seg.get("label")
    for a in range(0, n - window + 1, stride):
        b = a + window
        seqs.append(ch[a:b])
        sums.append(window_summary(ch[a:b], turn[a:b], u[a:b], v[a:b]))
        ys.append(int(lab[a:b].mean() >= min_anom_frac) if lab is not None else 0)
        t0.append(seg["t"][a])
    return {
        "seq": slog(np.asarray(seqs, np.float32)),
        "summary": slog(np.asarray(sums, np.float32)),
        "y": np.asarray(ys, np.int8),
        "t0": np.asarray(t0, np.float32),
    }


def merge(parts):
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
