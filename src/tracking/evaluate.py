"""Tracking evaluation on the 20 DUT Anti-UAV videos.

Each video has exactly one drone and one GT box per frame (-100 = absent).
We report MOT metrics (MOTA, IDF1, ID switches) with motmetrics plus the
single-object success rate (IoU > 0.5) commonly used for anti-UAV tracking.
The raw tracks are saved to reports/dut_tracks_videoNN.csv and then turned
into trajectory windows by src.trajectory.video.

    python -m src.tracking.evaluate
"""
import json

import numpy as np

if not hasattr(np, "asfarray"):  # motmetrics 1.4 still calls the removed NumPy 1.x helper
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

import motmetrics as mm  # noqa: E402
import pandas as pd

from src.config import DUT_RAW, MODELS, REPORTS
from src.tracking.track import track
from src.trajectory import video as video_traj



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


def main(imgsz=1280, conf=0.1):
    weights = MODELS / "detector" / "best.pt"
    frames_root = next(p for p in (DUT_RAW / "tracking").rglob("video01") if p.is_dir()).parent
    videos = sorted(p.name for p in frames_root.iterdir() if p.is_dir() and p.name.startswith("video"))

    acc_all, names, rows = [], [], []
    for v in videos:
        gt = load_gt(v)
        df = track(frames_root / v, weights, conf=conf, imgsz=imgsz)
        df.to_csv(REPORTS / f"dut_tracks_{v}.csv", index=False)

        acc = mm.MOTAccumulator(auto_id=True)
        hits = 0
        visible = 0
        for f, g in enumerate(gt):
            present = g[0] > -50
            gt_ids, gt_boxes = ([1], [[g[0], g[1], g[2], g[3]]]) if present else ([], [])
            pr = df[df.frame == f]
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
        acc_all.append(acc)
        names.append(v)
        rows.append({"video": v, "frames": len(gt), "visible": visible,
                     "success@0.5": hits / max(visible, 1), "tracks": int(df.track_id.nunique()) if len(df) else 0})

        print(rows[-1])

    mh = mm.metrics.create()
    summ = mh.compute_many(acc_all, names=names, generate_overall=True,
                           metrics=["mota", "motp", "idf1", "num_switches", "precision", "recall",
                                    "mostly_tracked", "num_fragmentations"])
    per = pd.DataFrame(rows).set_index("video")
    out = summ.join(per, how="left")
    out.to_csv(REPORTS / "tracking_metrics.csv")
    print(out.round(3))

    overall = summ.loc["OVERALL"].to_dict()
    overall["mean_success@0.5"] = float(per["success@0.5"].mean())
    (REPORTS / "tracking_summary.json").write_text(json.dumps(overall, indent=2, default=float))

    video_traj.main()


if __name__ == "__main__":
    main()
