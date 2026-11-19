"""Fine-tune YOLO11 on the merged drone dataset, then evaluate on the test
split (overall and per source). Also trains the drone-model classifier.

    python -m src.detection.train --epochs 60 --imgsz 960 --batch 8
    python -m src.detection.train --eval-only
"""
import argparse
import json
import shutil

from ultralytics import YOLO

from src.config import MODELS, PROCESSED, REPORTS, YOLO_DATA

RUNS = MODELS / "runs"
DET_DIR = MODELS / "detector"
CLS_DIR = MODELS / "classifier"


def subset_yaml(prefix, name):
    """YOLO yaml whose test split only holds images whose name starts with prefix."""
    imgs = sorted((YOLO_DATA / "images" / "test").glob(f"{prefix}*"))
    lst = YOLO_DATA / f"test_{name}.txt"
    lst.write_text("\n".join(p.as_posix() for p in imgs))
    y = YOLO_DATA / f"drone_{name}.yaml"
    y.write_text(f"path: {YOLO_DATA.as_posix()}\ntrain: images/train\nval: images/val\n"
                 f"test: {lst.as_posix()}\nnames:\n  0: drone\n")
    return y, len(imgs)


def evaluate(weights, imgsz):
    model = YOLO(str(weights))
    out = {}
    sets = {"all": (YOLO_DATA / "drone.yaml", None)}
    for prefix, name in (("dut_", "dut"), ("vd_", "visiodect")):
        y, n = subset_yaml(prefix, name)
        if n:
            sets[name] = (y, n)
    for name, (yaml_path, _) in sets.items():
        m = model.val(data=str(yaml_path), split="test", imgsz=imgsz, batch=8, device=0,
                      conf=0.001, iou=0.6, plots=(name == "all"),
                      project=str(RUNS), name=f"test_{name}", exist_ok=True, verbose=False)
        p, r = float(m.box.mp), float(m.box.mr)
        out[name] = {
            "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
            "images": int(m.seen) if hasattr(m, "seen") else None,
        }
        print(name, out[name])
    (REPORTS / "detection_metrics.json").write_text(json.dumps(out, indent=2))
    return out


def train_detector(a):
    last = RUNS / "detector" / "weights" / "last.pt"
    if a.resume and last.exists():
        YOLO(str(last)).train(resume=True)
        DET_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(RUNS / "detector" / "weights" / "best.pt", DET_DIR / "best.pt")
        return DET_DIR / "best.pt"
    model = YOLO(a.model)
    model.train(
        data=str(YOLO_DATA / "drone.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        device=0, workers=a.workers, project=str(RUNS), name="detector", exist_ok=True,
        patience=15, cos_lr=True, close_mosaic=10, amp=True, seed=0,
        degrees=5.0, scale=0.5, fliplr=0.5, mosaic=1.0, hsv_v=0.4,
        plots=True,
    )
    DET_DIR.mkdir(parents=True, exist_ok=True)
    best = RUNS / "detector" / "weights" / "best.pt"
    shutil.copy(best, DET_DIR / "best.pt")
    return DET_DIR / "best.pt"


def train_classifier(a):
    data = PROCESSED / "drone_cls"
    if not data.exists():
        print("no crop dataset, skipping classifier")
        return
    try:
        model = YOLO("yolo11n-cls.pt")
    except Exception as e:  # no ImageNet weights (offline): reuse the drone detector's backbone
        print("could not get yolo11n-cls.pt:", e)
        model = YOLO("yolo11s-cls.yaml").load(str(DET_DIR / "best.pt"))
    model.train(data=str(data), epochs=a.cls_epochs, imgsz=128, batch=128, device=0,
                workers=a.workers, project=str(RUNS), name="classifier", exist_ok=True,
                patience=8, seed=0)
    CLS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(RUNS / "classifier" / "weights" / "best.pt", CLS_DIR / "best.pt")
    m = YOLO(str(CLS_DIR / "best.pt")).val(data=str(data), split="test", imgsz=128, device=0,
                                         project=str(RUNS), name="classifier_test", exist_ok=True)
    res = {"top1": float(m.top1), "top5": float(m.top5)}
    print("classifier test", res)
    (REPORTS / "classifier_metrics.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cls-epochs", type=int, default=25)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue from runs/detector/weights/last.pt")
    ap.add_argument("--skip-classifier", action="store_true")
    ap.add_argument("--classifier-only", action="store_true")
    a = ap.parse_args()
    if a.classifier_only:
        train_classifier(a)
        raise SystemExit
    w = DET_DIR / "best.pt" if a.eval_only else train_detector(a)
    evaluate(w, a.imgsz)
    if not a.eval_only and not a.skip_classifier:
        train_classifier(a)
