"""Synthetic 2D drone trajectories with labelled anomalies.

Units are body lengths (scale = 1) at TRAJ_HZ. Positions get the kind of
jitter a detector + tracker adds, so the same features behave like they do on
real video tracks.

Normal behaviour  : cruise, gentle arcs, waypoint legs with smooth turns,
                    hover with drift, steady climb / descent.
Anomaly types     : zigzag, dive, speed_burst, reversal, circling, jump.
Each anomalous track is a normal track with one or two anomaly intervals
spliced in, and only those samples are labelled 1.
"""
import numpy as np

from src.config import TRAJ_HZ

ANOMALY_TYPES = ["zigzag", "dive", "speed_burst", "reversal", "circling", "jump"]


def _smooth_noise(rng, n, sigma, corr):
    """Low-pass random walk, used for slowly varying speed / heading."""
    x = rng.normal(0, sigma, n)
    k = max(1, int(corr * TRAJ_HZ))
    return np.convolve(x, np.ones(k) / np.sqrt(k), mode="same")


def normal_velocity(rng, n):
    """Return (vx, vy) arrays for one normal behaviour, body lengths / s."""
    dt = 1.0 / TRAJ_HZ
    kind = rng.choice(["cruise", "arc", "waypoints", "hover", "climb"],
                      p=[0.25, 0.2, 0.3, 0.15, 0.1])
    heading0 = rng.uniform(-np.pi, np.pi)

    if kind == "hover":
        vx = _smooth_noise(rng, n, 0.15, 2.0)
        vy = _smooth_noise(rng, n, 0.15, 2.0)
        return vx, vy, kind

    speed = rng.uniform(2, 15) + _smooth_noise(rng, n, 0.3, 3.0)
    speed = np.clip(speed, 0.5, None)

    if kind == "cruise":
        heading = heading0 + _smooth_noise(rng, n, 0.01, 3.0)
    elif kind == "arc":
        rate = rng.uniform(-0.25, 0.25)
        heading = heading0 + rate * np.arange(n) * dt
    elif kind == "climb":
        heading = rng.choice([1, -1]) * rng.uniform(0.4, 1.2) + _smooth_noise(rng, n, 0.01, 3.0)
        speed = np.clip(speed * 0.5, 0.5, None)
    else:  # waypoints: straight legs, slow down into each turn
        heading = np.empty(n)
        speed_mult = np.ones(n)
        h, i = heading0, 0
        while i < n:
            leg = int(rng.uniform(4, 10) * TRAJ_HZ)
            heading[i:i + leg] = h
            i += leg
            turn_len = int(rng.uniform(1.5, 3) * TRAJ_HZ)
            dh = rng.uniform(-np.pi / 2, np.pi / 2)
            seg = slice(i, min(i + turn_len, n))
            m = seg.stop - seg.start
            if m > 0:
                heading[seg] = h + dh * (1 - np.cos(np.linspace(0, np.pi, m))) / 2
                speed_mult[seg] = 0.6 + 0.4 * np.cos(np.linspace(-np.pi, np.pi, m)) ** 2
            h += dh
            i += turn_len
        speed = speed * speed_mult

    return speed * np.cos(heading), speed * np.sin(heading), kind


def inject(rng, vx, vy, pos, kind):
    """Modify velocities (or positions for 'jump') over a random interval.
    Returns a boolean label array."""
    n = len(vx)
    dur = int(rng.uniform(1.5, 4.0) * TRAJ_HZ)
    a = rng.integers(int(0.15 * n), max(int(0.15 * n) + 1, n - dur - 5))
    b = min(n, a + dur)
    lab = np.zeros(n, bool)
    lab[a:b] = True
    tt = np.arange(b - a) / TRAJ_HZ
    spd = max(np.hypot(vx[a], vy[a]), 2.0)

    if kind == "zigzag":
        f = rng.uniform(0.8, 2.5)
        amp = rng.uniform(1.0, 2.5) * spd
        h = np.arctan2(vy[a], vx[a]) + np.pi / 2
        lat = amp * np.sign(np.sin(2 * np.pi * f * tt))
        vx[a:b] += lat * np.cos(h)
        vy[a:b] += lat * np.sin(h)
    elif kind == "dive":
        vy[a:b] -= rng.uniform(12, 30) * np.sin(np.linspace(0, np.pi, b - a)) ** 0.3
    elif kind == "speed_burst":
        k = rng.uniform(3, 6)
        env = np.ones(b - a)
        env[:2] = [0.5, 0.8]
        vx[a:b] *= 1 + (k - 1) * env
        vy[a:b] *= 1 + (k - 1) * env
        if spd < 3:
            vx[a:b] += rng.uniform(10, 20) * np.cos(rng.uniform(-np.pi, np.pi))
    elif kind == "reversal":
        # snap back and forth: abrupt 180 degree turns
        flips = np.sign(np.sin(2 * np.pi * rng.uniform(0.3, 0.7) * tt + 1e-3))
        vx[a:b] *= flips * rng.uniform(1.0, 1.5)
        vy[a:b] *= flips * rng.uniform(1.0, 1.5)
    elif kind == "circling":
        r = rng.uniform(1.5, 4.0)
        w = rng.uniform(1.5, 3.0) * rng.choice([1, -1])
        ph = w * tt
        vx[a:b] = -r * w * np.sin(ph)
        vy[a:b] = r * w * np.cos(ph)
    elif kind == "jump":
        jumps = rng.integers(2, 5)
        for j in rng.choice(np.arange(a, b), size=min(jumps, b - a), replace=False):
            pos[j:, 0] += rng.normal(0, 6)
            pos[j:, 1] += rng.normal(0, 6)
    return lab


def make_track(rng, duration_s=None, anomaly=None, noise=None):
    n = int((duration_s or rng.uniform(15, 40)) * TRAJ_HZ)
    vx, vy, kind = normal_velocity(rng, n)
    vx, vy = vx.copy(), vy.copy()
    pos_offset = np.zeros((n, 2))
    label = np.zeros(n, bool)
    atype = np.zeros(n, np.int8)       # 0 = none, k+1 = ANOMALY_TYPES[k]
    if anomaly:
        types = [anomaly] if isinstance(anomaly, str) else anomaly
        for k in types:
            lab = inject(rng, vx, vy, pos_offset, k)
            label |= lab
            atype[lab] = ANOMALY_TYPES.index(k) + 1

    dt = 1.0 / TRAJ_HZ
    u = np.cumsum(vx) * dt + pos_offset[:, 0]
    v = np.cumsum(vy) * dt + pos_offset[:, 1]
    sigma = noise if noise is not None else rng.uniform(0.02, 0.12)
    u += rng.normal(0, sigma, n)
    v += rng.normal(0, sigma, n)
    return {
        "t": np.arange(n) * dt,
        "u": u, "v": v,
        "scale": np.ones(n),
        "label": label,
        "atype": atype,
        "behaviour": kind,
        "anomaly": anomaly or "none",
    }


def make_dataset(n_tracks, anomaly_rate, seed):
    rng = np.random.default_rng(seed)
    tracks = []
    for _ in range(n_tracks):
        if rng.random() < anomaly_rate:
            k = list(rng.choice(ANOMALY_TYPES, size=rng.integers(1, 3), replace=False))
            tracks.append(make_track(rng, anomaly=k[0] if len(k) == 1 else k))
        else:
            tracks.append(make_track(rng))
    return tracks


def to_windows(tracks, prefix):
    from src.config import STRIDE, WINDOW
    from src.trajectory.features import merge, windows_from_segment

    names = np.array(["none"] + ANOMALY_TYPES)
    parts = []
    for i, tr in enumerate(tracks):
        w = windows_from_segment(tr)
        if w is None:
            continue
        m = len(w["y"])
        cls = []
        for j, a in enumerate(range(0, len(tr["t"]) - WINDOW + 1, STRIDE)):
            at = tr["atype"][a:a + WINDOW]
            at = at[at > 0]
            cls.append(names[np.bincount(at).argmax()] if (w["y"][j] and len(at)) else "none")
        w["group"] = np.array([f"{prefix}{i}"] * m)
        w["cls"] = np.array(cls)
        parts.append(w)
    return merge(parts)


def main():
    import pickle

    from src.config import SEED, TRAJ_DATA

    sizes = {"train": (3000, 0.3), "val": (800, 0.4), "test": (1000, 0.4)}
    for i, (name, (n, rate)) in enumerate(sizes.items()):
        tracks = make_dataset(n, rate, SEED + i)
        d = to_windows(tracks, f"syn_{name}_")
        np.savez_compressed(TRAJ_DATA / f"synthetic_{name}.npz", **d)
        if name == "test":
            with open(TRAJ_DATA / "synthetic_test_tracks.pkl", "wb") as f:
                pickle.dump(tracks[:60], f)
        print(f"{name}: {len(d['y'])} windows, {int(d['y'].sum())} anomalous")


if __name__ == "__main__":
    main()
