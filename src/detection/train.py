"""Fine-tune YOLO11 on the merged drone dataset, then evaluate on the test
split (overall and per source). Optionally runs the drone-model classifier
evaluation afterwards (src.detection.classifier_eval).

    python -m src.detection.train --data block --name detector_block --epochs 20 --imgsz 960
    python -m src.detection.train --data block --eval-only

--data random  original dataset (VisioDECT split image by image, leaky)
--data block   VisioDECT split by frame blocks (src.detection.block_split)
"""
import argparse
import json
import shutil

from ultralytics import YOLO

from src.config import MODELS, PROCESSED, REPORTS, YOLO_DATA

RUNS = MODELS / "runs"
DET_DIR = MODELS / "detector"
DATASETS = {"random": YOLO_DATA, "block": PROCESSED / "yolo_block"}


def subset_yaml(root, prefix, name):
    """YOLO yaml whose test split only holds images whose name starts with prefix."""
    imgs = sorted((root / "images" / "test").glob(f"{prefix}*"))
    lst = root / f"test_{name}.txt"
    lst.write_text("\n".join(p.as_posix() for p in imgs))
    y = root / f"drone_{name}.yaml"
    y.write_text(f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\n"
                 f"test: {lst.as_posix()}\nnames:\n  0: drone\n")
    return y, len(imgs)


def metrics_file(data):
    return REPORTS / ("detection_metrics.json" if data == "random" else f"detection_metrics_{data}.json")


def evaluate(weights, imgsz, data="random"):
    root = DATASETS[data]
    model = YOLO(str(weights))
    out = {}
    sets = {"all": root / "drone.yaml"}
    for prefix, name in (("dut_", "dut"), ("vd_", "visiodect")):
        y, n = subset_yaml(root, prefix, name)
        if n:
            sets[name] = y
    for name, yaml_path in sets.items():
        m = model.val(data=str(yaml_path), split="test", imgsz=imgsz, batch=8, device=0,
                      conf=0.001, iou=0.6, plots=(name == "all"),
                      project=str(RUNS), name=f"test_{data}_{name}", exist_ok=True, verbose=False)
        p, r = float(m.box.mp), float(m.box.mr)
        out[name] = {
            "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
        }
        print(name, out[name])
    metrics_file(data).write_text(json.dumps(out, indent=2))
    return out


def install_detector(run_name):
    """Copy the run's best weights to models/detector/best.pt. The detector
    installed before is kept as best_<its run name>.pt."""
    DET_DIR.mkdir(parents=True, exist_ok=True)
    cur = DET_DIR / "best.pt"
    tag = DET_DIR / "installed_from.txt"
    old = tag.read_text().strip() if tag.exists() else "detector"
    if cur.exists() and old != run_name:
        shutil.copy(cur, DET_DIR / f"best_{old}.pt")
    shutil.copy(RUNS / run_name / "weights" / "best.pt", cur)
    tag.write_text(run_name)
    return cur


def train_detector(a):
    last = RUNS / a.name / "weights" / "last.pt"
    if a.resume and last.exists():
        YOLO(str(last)).train(resume=True)
        return install_detector(a.name)
    model = YOLO(a.model)
    model.train(
        data=str(DATASETS[a.data] / "drone.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        device=0, workers=a.workers, project=str(RUNS), name=a.name, exist_ok=True,
        patience=15, cos_lr=True, close_mosaic=10, amp=True, seed=0,
        degrees=5.0, scale=0.5, fliplr=0.5, mosaic=1.0, hsv_v=0.4,
        plots=True,
    )
    return install_detector(a.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data", choices=list(DATASETS), default="block")
    ap.add_argument("--name", default="detector_block", help="run name under models/runs")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue from runs/<name>/weights/last.pt")
    ap.add_argument("--with-classifier", action="store_true",
                    help="also run src.detection.classifier_eval afterwards")
    a = ap.parse_args()
    w = DET_DIR / "best.pt" if a.eval_only else train_detector(a)
    evaluate(w, a.imgsz, a.data)
    if a.with_classifier:
        from src.detection import classifier_eval
        classifier_eval.main(workers=a.workers)
