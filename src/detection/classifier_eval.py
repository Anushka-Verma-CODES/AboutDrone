"""Honest evaluation of the drone-model classifier.

VisioDECT images are consecutive video frames: for each drone model and
weather scenario there are 1200 numbered frames, and neighbouring frames are
near-duplicates. A random image split therefore puts almost the same picture
in train and test and over-states accuracy.

Two leakage-free protocols, both built from the existing crops in
data/processed/drone_cls (hard links, no copies):

- block:  every (model, scenario) sequence is cut into 12 blocks of 100
          frames; 8 blocks train, 2 val, 2 test. The first and last 5 frames
          of every block are dropped so no test frame has a near-identical
          neighbour in training.
- cross-weather: train on two weather scenarios, test on the third
          (leave-one-scenario-out, 3 folds).

The random-split model is retrained with the same settings for a fair
comparison. The block-split model becomes models/classifier/best.pt.

    python -m src.detection.classifier_eval
"""
import json
import os
import re
import shutil
from collections import Counter, defaultdict

import numpy as np
from ultralytics import YOLO

from src.config import MODELS, PROCESSED, REPORTS, SEED

CROPS = PROCESSED / "drone_cls"
SPLITS = PROCESSED / "cls_splits"
RUNS = MODELS / "runs"
BASE = "yolo11s-cls.pt"           # ImageNet-pretrained
NAME_RE = re.compile(r"^vd_(?P<model>.+?)_(?P<scen>cloudy|evening|sunny)_.*?_(?P<num>\d+)_\d+\.jpg$", re.I)
BLOCK, EDGE = 100, 5
SCENARIOS = ("cloudy", "evening", "sunny")


def all_crops():
    out = []
    for p in CROPS.glob("*/*/*.jpg"):
        m = NAME_RE.match(p.name)
        if m:
            out.append({"path": p, "cls": p.parent.name, "scen": m["scen"].lower(), "num": int(m["num"])})
    return out


def block_of(num):
    i = num - 1
    return i // BLOCK, i % BLOCK


def assign_blocks(crops, seed=SEED):
    """(cls, scen, block) -> split, 8/2/2 blocks per sequence."""
    rng = np.random.default_rng(seed)
    blocks = defaultdict(set)
    for c in crops:
        blocks[(c["cls"], c["scen"])].add(block_of(c["num"])[0])
    out = {}
    for key, bs in sorted(blocks.items()):
        bs = list(rng.permutation(sorted(bs)))
        n_test = n_val = max(1, round(len(bs) / 6))
        for j, b in enumerate(bs):
            out[key + (b,)] = "test" if j < n_test else "val" if j < n_test + n_val else "train"
    return out


def materialise(name, items):
    """items: iterable of (crop, split). Hard-links crops into SPLITS/name."""
    root = SPLITS / name
    if root.exists():
        shutil.rmtree(root)
    n = Counter()
    for c, split in items:
        dst = root / split / c["cls"] / c["path"].name
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(c["path"], dst)
        except OSError:
            shutil.copy2(c["path"], dst)
        n[split] += 1
    return root, dict(n)


def block_split(crops):
    blocks = assign_blocks(crops)
    for c in crops:
        b, off = block_of(c["num"])
        if EDGE <= off < BLOCK - EDGE:
            yield c, blocks[(c["cls"], c["scen"], b)]


def cross_weather_split(crops, held_out):
    """Test = one weather scenario; val = held-out blocks of the other two."""
    blocks = assign_blocks(crops)
    for c in crops:
        if c["scen"] == held_out:
            yield c, "test"
            continue
        b, off = block_of(c["num"])
        if EDGE <= off < BLOCK - EDGE:
            yield c, "val" if blocks[(c["cls"], c["scen"], b)] == "val" else "train"


def random_split(crops):
    for c in crops:
        yield c, c["path"].parent.parent.name     # the original split folder


def train_eval(name, data, epochs, workers):
    model = YOLO(BASE)
    model.train(data=str(data), epochs=epochs, imgsz=128, batch=128, device=0, workers=workers,
                project=str(RUNS), name=f"cls_{name}", exist_ok=True, patience=8, seed=0,
                plots=False, verbose=False)
    best = RUNS / f"cls_{name}" / "weights" / "best.pt"
    m = YOLO(str(best)).val(data=str(data), split="test", imgsz=128, batch=128, device=0,
                            project=str(RUNS), name=f"cls_{name}_test", exist_ok=True,
                            plots=True, verbose=False)
    res = {"top1": float(m.top1), "top5": float(m.top5)}
    print(name, res)
    return best, res


def main(epochs=20, workers=6):
    crops = all_crops()
    print("crops:", len(crops), Counter(c["scen"] for c in crops))
    results = {}

    root, n = materialise("random", random_split(crops))
    _, results["random_split"] = train_eval("random", root, epochs, workers)
    results["random_split"]["sizes"] = n

    root, n = materialise("block", block_split(crops))
    best, results["block_split"] = train_eval("block", root, epochs, workers)
    results["block_split"]["sizes"] = n
    (MODELS / "classifier").mkdir(parents=True, exist_ok=True)
    shutil.copy(best, MODELS / "classifier" / "best.pt")

    results["cross_weather"] = {}
    for scen in SCENARIOS:
        root, n = materialise(f"loso_{scen}", cross_weather_split(crops, scen))
        _, r = train_eval(f"loso_{scen}", root, epochs, workers)
        r["sizes"] = n
        results["cross_weather"][f"test_{scen}"] = r
    cw = results["cross_weather"]
    cw["mean_top1"] = float(np.mean([v["top1"] for k, v in cw.items() if k.startswith("test_")]))

    (REPORTS / "classifier_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
