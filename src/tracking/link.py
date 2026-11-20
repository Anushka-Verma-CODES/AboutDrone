"""Offline tracklet linking.

When a small drone is missed for a few frames (it shrinks, passes clutter,
or the detector confidence dips), the online tracker often starts a new ID
when it reappears. Afterwards we can join such pieces: tracklet B is linked
to tracklet A when

- B starts after A ends, within MAX_GAP_S seconds,
- B's first position is close to where A would be at that time, predicted
  with A's recent velocity (tolerance grows with the gap, in box sizes),
- the box sizes are similar.

Pairs are joined greedily from the most to the least consistent. Tracks
that overlap in time are never joined.

MAX_GAP_S was picked on the 12 anomaly-training DUT videos (IDF1 0.637 raw,
0.684 / 0.735 / 0.791 with 2 / 4 / 8 s). DUT has one drone per video, which
makes long gaps look free; 8 s is kept as a cap so that different drones in
multi-drone footage are less likely to be merged.
"""
import numpy as np
import pandas as pd

MAX_GAP_S = 8.0       # seconds
TAIL = 10             # frames used to estimate the end velocity
BASE_TOL = 1.5        # box sizes
TOL_PER_FRAME = 0.15  # extra box sizes per missing frame
MAX_SIZE_RATIO = 2.0


def _summary(g):
    g = g.sort_values("frame")
    f = g.frame.to_numpy()
    cx = ((g.x1 + g.x2) / 2).to_numpy()
    cy = ((g.y1 + g.y2) / 2).to_numpy()
    size = np.sqrt(np.clip((g.x2 - g.x1) * (g.y2 - g.y1), 1, None)).to_numpy()
    tail = slice(max(0, len(f) - TAIL), len(f))
    if len(f) >= 3:
        vx = np.polyfit(f[tail], cx[tail], 1)[0] if len(f[tail]) >= 2 else 0.0
        vy = np.polyfit(f[tail], cy[tail], 1)[0] if len(f[tail]) >= 2 else 0.0
    else:
        vx = vy = 0.0
    return {"start": f[0], "end": f[-1], "p0": (cx[0], cy[0]), "p1": (cx[-1], cy[-1]),
            "v": (vx, vy), "s0": np.median(size[:TAIL]), "s1": np.median(size[tail])}


def link_tracks(df, fps=25.0, max_gap_s=MAX_GAP_S, base_tol=BASE_TOL, tol_per_frame=TOL_PER_FRAME):
    """Returns a copy of df with linked tracklets sharing one track_id, and
    the number of links made."""
    max_gap = int(round(max_gap_s * fps))
    if df.empty:
        return df.copy(), 0
    info = {tid: _summary(g) for tid, g in df.groupby("track_id")}
    cand = []
    for a, A in info.items():
        for b, B in info.items():
            gap = B["start"] - A["end"]
            if a == b or gap <= 0 or gap > max_gap:
                continue
            ratio = max(A["s1"], B["s0"]) / max(min(A["s1"], B["s0"]), 1e-6)
            if ratio > MAX_SIZE_RATIO:
                continue
            pred = (A["p1"][0] + A["v"][0] * gap, A["p1"][1] + A["v"][1] * gap)
            size = 0.5 * (A["s1"] + B["s0"])
            d = np.hypot(pred[0] - B["p0"][0], pred[1] - B["p0"][1]) / max(size, 1.0)
            if d <= base_tol + tol_per_frame * gap:
                cand.append((d + 0.01 * gap, a, b))
    cand.sort()
    tail_used, head_used, parent = set(), set(), {}

    def root(x):
        while x in parent:
            x = parent[x]
        return x

    links = 0
    for _, a, b in cand:
        if a in tail_used or b in head_used or root(a) == root(b):
            continue
        tail_used.add(a)
        head_used.add(b)
        parent[b] = a
        links += 1
    out = df.copy()
    out["track_id"] = out.track_id.map(lambda t: root(t))
    # a merged track could in principle hold two boxes in one frame; keep the most confident
    out = out.sort_values("conf", ascending=False).drop_duplicates(["frame", "track_id"])
    return out.sort_values(["frame", "track_id"]).reset_index(drop=True), links


# Tracks whose most confident detection stays below this are dropped: on the
# 12 anomaly-training DUT videos (BoT-SORT + linking) MOTA goes 0.693 -> 0.837
# and IDF1 0.816 -> 0.878 at 0.75 with recall 0.906 -> 0.903; real drone tracks
# start to be lost only above 0.85. Lower it for footage with very distant drones.
MIN_TRACK_CONF = 0.75


def filter_tracks(df, min_conf=MIN_TRACK_CONF):
    """Drop tracks that never reach min_conf (mostly short false positives)."""
    if df.empty:
        return df.copy()
    best = df.groupby("track_id").conf.max()
    return df[df.track_id.isin(best[best >= min_conf].index)].reset_index(drop=True)
