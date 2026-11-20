"""Tracking evaluation on the 20 DUT Anti-UAV videos.

Each video has exactly one drone and one GT box per frame (-100 = absent).
We report MOT metrics (MOTA, IDF1, ID switches) with motmetrics plus the
single-object success rate (IoU > 0.5) commonly used for anti-UAV tracking.

Six configurations are compared: ByteTrack and BoT-SORT (with global motion
compensation), each raw, with offline tracklet linking, and with linking plus
a track-confidence filter (src.tracking.link). The configuration used downstream is the one with the
best IDF1 on the anomaly *training* videos only (src.trajectory.video split),
so the test videos do not influence the choice.

Outputs
    reports/tracking_comparison.csv       every configuration, train/val/test/all videos
    reports/tracking_metrics.csv          per video, chosen configuration
    reports/tracking_summary.json         overall, chosen configuration
    reports/dut_tracks_videoNN.csv        chosen tracks, input to src.trajectory.video

    python -m src.tracking.evaluate            # run trackers (GPU) + evaluate
    python -m src.tracking.evaluate --reuse    # reuse saved raw tracks
"""
import argparse
import json

import numpy as np

if not hasattr(np, "asfarray"):  # motmetrics 1.4 still calls the removed NumPy 1.x helper
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

import motmetrics as mm  # noqa: E402
import pandas as pd

from src.config import DUT_RAW, MODELS, REPORTS
from src.tracking.link import filter_tracks, link_tracks
from src.tracking.track import TRACKERS, track
from src.trajectory import video as video_traj

RAW_DIR = REPORTS / "tracks"
MOT_METRICS = ["mota", "motp", "idf1", "num_switches", "precision", "recall",
               "mostly_tracked", "num_fragmentations"]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(video):
    gt_dir = DUT_RAW / "tracking"
    f = next(gt_dir.rglob(f"{video}_gt.txt"))
    rows = [list(map(float, l.replace(",", " ").split())) for l in f.read_text().splitlines() if l.strip()]
    return np.array(rows)


def score_video(df, gt):
    """motmetrics accumulator + single-object success rate for one video."""
    acc = mm.MOTAccumulator(auto_id=True)
    by_frame = {f: g for f, g in df.groupby("frame")} if len(df) else {}
    hits = visible = 0
    for f, g in enumerate(gt):
        present = g[0] > -50
        gt_ids, gt_boxes = ([1], [[g[0], g[1], g[2], g[3]]]) if present else ([], [])
        pr = by_frame.get(f, df.iloc[:0])
        hyp_ids = pr.track_id.tolist()
        hyp_boxes = [[r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1] for r in pr.itertuples()]
        dist = mm.distances.iou_matrix(gt_boxes, hyp_boxes, max_iou=0.5) if gt_boxes and hyp_boxes \
            else np.empty((len(gt_ids), len(hyp_ids)))
        acc.update(gt_ids, hyp_ids, dist)
        if present:
            visible += 1
            gb = (g[0], g[1], g[0] + g[2], g[1] + g[3])
            if len(pr) and max(iou(gb, (r.x1, r.y1, r.x2, r.y2)) for r in pr.itertuples()) > 0.5:
                hits += 1
    return acc, {"frames": len(gt), "visible": visible, "success@0.5": hits / max(visible, 1),
                 "tracks": int(df.track_id.nunique()) if len(df) else 0}


def summarise(accs, names, rows):
    mh = mm.metrics.create()
    summ = mh.compute_many(accs, names=names, generate_overall=True, metrics=MOT_METRICS)
    per = pd.DataFrame(rows, index=names)
    overall = summ.loc["OVERALL"].to_dict()
    overall["mean_success@0.5"] = float(per["success@0.5"].mean())
    overall["tracks_per_video"] = float(per["tracks"].mean())
    return summ.join(per, how="left"), overall


def main(reuse=False, imgsz=1280, conf=0.1):
    weights = MODELS / "detector" / "best.pt"
    frames_root = next(p for p in (DUT_RAW / "tracking").rglob("video01") if p.is_dir()).parent
    videos = sorted(p.name for p in frames_root.iterdir() if p.is_dir() and p.name.startswith("video"))
    split = video_traj.video_split(videos)
    split["all"] = videos
    gts = {v: load_gt(v) for v in videos}

    tracks = {}
    for trk in TRACKERS:
        (RAW_DIR / trk).mkdir(parents=True, exist_ok=True)
        for v in videos:
            p = RAW_DIR / trk / f"{v}.csv"
            if reuse and p.exists():
                df = pd.read_csv(p)
            else:
                df = track(frames_root / v, weights, conf=conf, imgsz=imgsz, tracker=trk)
                df.to_csv(p, index=False)
                print(trk, v, "tracks:", df.track_id.nunique() if len(df) else 0)
            tracks[(trk, "raw", v)] = df
            tracks[(trk, "linked", v)] = link_tracks(df)[0]
            tracks[(trk, "linked+filtered", v)] = filter_tracks(tracks[(trk, "linked", v)])

    rows, per_cfg = [], {}
    for trk in TRACKERS:
        for linked in ("raw", "linked", "linked+filtered"):
            res = {v: score_video(tracks[(trk, linked, v)], gts[v]) for v in videos}
            per_cfg[(trk, linked)] = res
            for part, vids in split.items():
                table, overall = summarise([res[v][0] for v in vids], list(vids), [res[v][1] for v in vids])
                rows.append({"tracker": trk, "post": linked, "videos": part, **overall})
    comp = pd.DataFrame(rows)
    comp.to_csv(REPORTS / "tracking_comparison.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(comp[["tracker", "post", "videos", "mota", "idf1", "num_switches",
                    "num_fragmentations", "mean_success@0.5", "tracks_per_video"]])

    train_rows = comp[comp.videos == "train"]
    best = train_rows.loc[train_rows.idf1.idxmax()]
    cfg = (best.tracker, best.post)
    print("chosen on training videos:", cfg)

    res = per_cfg[cfg]
    table, overall = summarise([res[v][0] for v in videos], videos, [res[v][1] for v in videos])
    table.to_csv(REPORTS / "tracking_metrics.csv")
    overall.update({"tracker": cfg[0], "post_processing": cfg[1]})
    (REPORTS / "tracking_summary.json").write_text(json.dumps(overall, indent=2, default=float))
    for v in videos:
        tracks[(cfg[0], cfg[1], v)].to_csv(REPORTS / f"dut_tracks_{v}.csv", index=False)

    video_traj.main()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="reuse raw tracks in reports/tracks/")
    main(reuse=ap.parse_args().reuse)
