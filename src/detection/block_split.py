"""Leakage-free YOLO dataset: VisioDECT re-split by frame blocks.

The first detector used a random 70/15/15 split of VisioDECT images. Those
are consecutive video frames, so near-identical frames ended up in train and
test and the VisioDECT score was optimistic. This builds data/processed/yolo_block
from the existing YOLO dataset (hard links, no copies):

- VisioDECT: frame-block split shared with the classifier (src.detection.blocks);
  frames at block edges are dropped.
- DUT: unchanged (DUT ships its own split of separate images).

    python -m src.detection.block_split
"""
import os
import re
import shutil
from collections import Counter

from src.config import PROCESSED, YOLO_DATA
from src.detection.blocks import assign_blocks, near_edge

YOLO_BLOCK = PROCESSED / "yolo_block"
NAME_RE = re.compile(r"^vd_(?P<model>.+?)_(?P<scen>cloudy|evening|sunny)_.*?_(?P<num>\d+)$", re.I)


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    imgs = sorted(YOLO_DATA.glob("images/*/*.jpg"))
    vd = []
    for p in imgs:
        m = NAME_RE.match(p.stem)
        if m:
            vd.append((p, (m["model"], m["scen"].lower()), int(m["num"])))
    blocks = assign_blocks((key, num) for _, key, num in vd)

    if YOLO_BLOCK.exists():
        shutil.rmtree(YOLO_BLOCK)
    n = Counter()
    vd_paths = {p for p, _, _ in vd}
    for p in imgs:
        if p in vd_paths:
            continue
        if p.stem.startswith("dut_"):
            split = p.parent.name
            link(p, YOLO_BLOCK / "images" / split / p.name)
            link(YOLO_DATA / "labels" / split / f"{p.stem}.txt", YOLO_BLOCK / "labels" / split / f"{p.stem}.txt")
            n[("dut", split)] += 1
    for p, key, num in vd:
        if near_edge(num):
            n[("visiodect", "dropped")] += 1
            continue
        split = blocks[key + ((num - 1) // 100,)]
        link(p, YOLO_BLOCK / "images" / split / p.name)
        link(YOLO_DATA / "labels" / p.parent.name / f"{p.stem}.txt", YOLO_BLOCK / "labels" / split / f"{p.stem}.txt")
        n[("visiodect", split)] += 1

    (YOLO_BLOCK / "drone.yaml").write_text(
        f"path: {YOLO_BLOCK.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: drone\n")
    for k in sorted(n):
        print(k, n[k])


if __name__ == "__main__":
    main()
