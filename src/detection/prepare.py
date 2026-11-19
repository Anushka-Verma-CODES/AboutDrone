"""Build one YOLO-format detection dataset (single class: drone) from
VisioDECT and DUT Anti-UAV, plus a crop dataset for drone-model
classification from VisioDECT.

    python -m src.detection.prepare

Output
    data/processed/yolo/{images,labels}/{train,val,test}/...
    data/processed/yolo/drone.yaml
    data/processed/drone_cls/{train,val,test}/<model>/*.jpg
"""
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm

from src.config import DUT_RAW, PROCESSED, SEED, VISIODECT_RAW, YOLO_DATA

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
CLS_DATA = PROCESSED / "drone_cls"


MAX_SIDE = 1280


def link_or_copy(src: Path, dst: Path, w: int, h: int):
    """Hard-link small images; downscale anything larger than MAX_SIDE so the
    data loader is not stuck decoding 1080p JPEGs every epoch. Labels are
    normalised, so they stay valid."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if max(w, h) > MAX_SIDE:
        im = cv2.imread(str(src))
        f = MAX_SIDE / max(w, h)
        im = cv2.resize(im, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst.with_suffix(".jpg")), im, [cv2.IMWRITE_JPEG_QUALITY, 93])
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        dst.write_bytes(src.read_bytes())


def write_label(path: Path, boxes, w, h):
    """boxes: list of (x1, y1, x2, y2) in pixels."""
    lines = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return len(lines)


def read_voc(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = int(float(size.find("width").text)), int(float(size.find("height").text))
    boxes = []
    for o in root.iter("object"):
        bb = o.find("bndbox")
        boxes.append(tuple(float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")))
    return w, h, boxes


# ------------------------------------------------------------------ DUT

def dut_items():
    """DUT ships its own train/val/test split, which we keep."""
    items = []
    base = DUT_RAW / "detection"
    for split in ("train", "val", "test"):
        for img in sorted((base / split / "img").glob("*.jpg")):
            xml = base / split / "xml" / f"{img.stem}.xml"
            if not xml.exists():
                continue
            w, h, boxes = read_voc(xml)
            items.append({"img": img, "boxes": boxes, "w": w, "h": h, "split": split,
                          "name": f"dut_{split}_{img.stem}", "model": None, "scenario": None})
    return items


# ------------------------------------------------------------------ VisioDECT

def visiodect_items():
    """Layout: <model>/images/<Scenario>/<name>.jpg and
    <model>/labels/<scenario>/voc/<name>.xml. Images without an annotation
    are skipped. Some XML files state a different size from the actual
    image, so boxes are rescaled to the real image size."""
    items = []
    base = VISIODECT_RAW / "extracted"
    model_dirs = [d for d in base.rglob("*") if d.is_dir() and (d / "images").is_dir() and (d / "labels").is_dir()]
    for mdir in model_dirs:
        model = mdir.name
        xmls = {p.stem.lower(): p for p in (mdir / "labels").rglob("*.xml")}
        imgs = [p for p in (mdir / "images").rglob("*") if p.suffix.lower() in IMG_EXT]
        for img in tqdm(imgs, desc=model):
            xml = xmls.get(img.stem.lower())
            if xml is None:
                continue
            try:
                with Image.open(img) as im:   # header only, no full decode
                    w, h = im.size
            except OSError:
                continue
            xw, xh, boxes = read_voc(xml)
            if (xw, xh) != (w, h) and xw > 0 and xh > 0:
                sx, sy = w / xw, h / xh
                boxes = [(a * sx, b * sy, c * sx, d * sy) for a, b, c, d in boxes]
            scenario = img.parent.name.lower()
            stem = img.stem.replace(" ", "").replace("(", "_").replace(")", "")
            items.append({"img": img, "boxes": boxes, "w": w, "h": h, "split": None,
                          "name": f"vd_{model}_{scenario}_{stem}",
                          "model": model, "scenario": scenario})
    # stratified 70/15/15 split by (model, scenario)
    groups = defaultdict(list)
    for it in items:
        groups[(it["model"], it["scenario"])].append(it)
    rng = random.Random(SEED)
    for g in groups.values():
        g.sort(key=lambda x: x["name"])
        rng.shuffle(g)
        n = len(g)
        for i, it in enumerate(g):
            it["split"] = "train" if i < 0.7 * n else "val" if i < 0.85 * n else "test"
    return items


def build_crops(items):
    """Square-padded crops around each VisioDECT box for model classification."""
    n = Counter()
    for it in tqdm(items, desc="crops"):
        im = None
        for k, (x1, y1, x2, y2) in enumerate(it["boxes"]):
            if im is None:
                im = cv2.imread(str(it["img"]))
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            s = max(x2 - x1, y2 - y1) * 1.3
            a, b = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
            c, d = int(min(it["w"], cx + s / 2)), int(min(it["h"], cy + s / 2))
            if c - a < 8 or d - b < 8:
                continue
            out = CLS_DATA / it["split"] / it["model"] / f"{it['name']}_{k}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                cv2.imwrite(str(out), im[b:d, a:c])
            n[(it["split"], it["model"])] += 1
    return n


def main():
    items = dut_items()
    vd = visiodect_items() if (VISIODECT_RAW / "extracted").exists() else []
    print(f"DUT images: {len(items)}   VisioDECT images: {len(vd)}")
    items += vd

    from concurrent.futures import ThreadPoolExecutor

    def one(it):
        link_or_copy(it["img"], YOLO_DATA / "images" / it["split"] / f"{it['name']}.jpg", it["w"], it["h"])
        return write_label(YOLO_DATA / "labels" / it["split"] / f"{it['name']}.txt",
                           it["boxes"], it["w"], it["h"])

    stats = Counter()
    with ThreadPoolExecutor(8) as ex:
        for it, nb in zip(items, tqdm(ex.map(one, items), total=len(items), desc="yolo")):
            src = "dut" if it["model"] is None else "visiodect"
            stats[(src, it["split"])] += 1
            stats[(src, "boxes")] += nb

    (YOLO_DATA / "drone.yaml").write_text(
        f"path: {YOLO_DATA.as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: drone\n")
    for k, v in sorted(stats.items()):
        print(k, v)

    if vd:
        n = build_crops(vd)
        for k, v in sorted(n.items()):
            print("cls", k, v)


if __name__ == "__main__":
    main()
