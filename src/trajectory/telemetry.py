"""Turn UAV-SEAD PX4 logs into labelled trajectory windows.

Each flight's estimated local position (vehicle_local_position) is projected
onto a vertical plane, the view a ground camera would have:
    u = x*cos(az) + y*sin(az)      horizontal, random camera azimuth
    v = -z                         altitude, up positive
Two azimuths per flight are used as augmentation.

Annotation ranges in mapping.json are PX4 timestamps (microseconds); any
sample inside a range is labelled anomalous.

    python -m src.trajectory.telemetry
"""
import json
from collections import Counter

import numpy as np
from pyulog import ULog
from tqdm import tqdm

from src.config import SEAD_RAW, SEED, TRAJ_DATA
from src.trajectory.features import merge, resample, windows_from_segment

FRAME_SIZE_M = 0.5
AZIMUTHS_PER_FLIGHT = 2


def load_flight(path):
    ulog = ULog(str(path), ["vehicle_local_position", "vehicle_land_detected"])
    pos = next(d for d in ulog.data_list if d.name == "vehicle_local_position")
    t = pos.data["timestamp"].astype(np.int64)
    x, y, z = pos.data["x"], pos.data["y"], pos.data["z"]

    airborne = np.ones(len(t), bool)
    land = [d for d in ulog.data_list if d.name == "vehicle_land_detected"]
    if land and "landed" in land[0].data:
        lt = land[0].data["timestamp"]
        landed = np.interp(t, lt, land[0].data["landed"].astype(float)) > 0.5
        airborne = ~landed
    ok = airborne & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return t[ok], x[ok], y[ok], z[ok]


def label_samples(t_us, annotation):
    lab = np.zeros(len(t_us), bool)
    for _field, ranges in annotation.get("ranges", []):
        for a, b in ranges:
            lab |= (t_us >= min(a, b)) & (t_us <= max(a, b))
    return lab


def split_flights(flights, classes, seed):
    """Stratified 60/20/20 split by flight so no flight leaks across splits."""
    rng = np.random.default_rng(seed)
    out = {}
    for c in set(classes.values()):
        fl = sorted(f for f in flights if classes[f] == c)
        rng.shuffle(fl)
        n = len(fl)
        a, b = int(0.6 * n), int(0.8 * n)
        out.update({f: "train" for f in fl[:a]})
        out.update({f: "val" for f in fl[a:b]})
        out.update({f: "test" for f in fl[b:]})
    return out


def main():
    mapping = json.loads((SEAD_RAW / "mapping.json").read_text())
    logs = sorted((SEAD_RAW / "ulg_files").rglob("*.ulg"))
    rel = {p: p.relative_to(SEAD_RAW / "ulg_files").with_suffix("").as_posix() for p in logs}
    logs = [p for p in logs if rel[p] in mapping]
    classes = {rel[p]: mapping[rel[p]]["annotations"][-1]["class"] for p in logs}
    split = split_flights(list(classes), classes, SEED)
    rng = np.random.default_rng(SEED)

    buckets = {"train": [], "val": [], "test": []}
    skipped = Counter()
    for p in tqdm(logs, desc="flights"):
        fid = rel[p]
        try:
            t, x, y, z = load_flight(p)
        except Exception:
            skipped["unreadable"] += 1
            continue
        if len(t) < 50:
            skipped["too_short"] += 1
            continue
        lab = label_samples(t, mapping[fid]["annotations"][-1])
        ts = (t - t[0]) / 1e6

        for k in range(AZIMUTHS_PER_FLIGHT):
            az = rng.uniform(0, 2 * np.pi)
            u = x * np.cos(az) + y * np.sin(az)
            segs = resample(ts, u, -z, np.full(len(t), FRAME_SIZE_M), label=lab)
            for s in segs:
                w = windows_from_segment(s)
                if w is None:
                    continue
                m = len(w["y"])
                w["group"] = np.array([f"{fid}#{k}"] * m)
                w["cls"] = np.array([classes[fid]] * m)
                buckets[split[fid]].append(w)

    for name, parts in buckets.items():
        d = merge(parts)
        np.savez_compressed(TRAJ_DATA / f"telemetry_{name}.npz", **d)
        print(f"{name}: {len(d['y'])} windows, {int(d['y'].sum())} anomalous, "
              f"{len(set(g.split('#')[0] for g in d['group']))} flights")
    if skipped:
        print("skipped:", dict(skipped))


if __name__ == "__main__":
    main()
