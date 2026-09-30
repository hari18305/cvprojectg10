"""Export our datasets in PASCAL VOC layout so the authors' C++ BING can read them.

Writes <out>/JPEGImages/<id>.jpg, Annotations/<id>.xml (standard VOC) and
Annotations/<id>.yml (the OpenCV YAML the C++ code reads), plus
ImageSets/Main/{train,test,class}.txt.  VOC boxes are 1-based and inclusive:
xmin = x1 + 1, xmax = x2 for our 0-based, end-exclusive boxes.

  python scripts/export_voc_format.py --out /tmp/voc_synth               # synthetic train/test
  python scripts/export_voc_format.py --out /tmp/voc_real --test real    # train synthetic, test real
"""
import argparse
import os
import shutil
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bing.datasets import ROOT, JsonDataset  # noqa: E402


def write_xml(path, name, shape, boxes, cats):
    H, W = shape[:2]
    objs = "".join(
        f"<object><name>{c}</name><difficult>0</difficult><bndbox>"
        f"<xmin>{int(round(b[0])) + 1}</xmin><ymin>{int(round(b[1])) + 1}</ymin>"
        f"<xmax>{int(round(b[2]))}</xmax><ymax>{int(round(b[3]))}</ymax></bndbox></object>"
        for b, c in zip(boxes, cats))
    with open(path, "w") as f:
        f.write(f"<annotation><filename>{name}</filename><size><width>{W}</width>"
                f"<height>{H}</height><depth>3</depth></size>{objs}</annotation>\n")


def write_yml(path, boxes, cats):
    """Same structure the authors' xml2yaml/yml.m tools produce (all values as strings)."""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_FORMAT_YAML)
    fs.startWriteStruct("annotation", cv2.FILE_NODE_MAP)
    fs.startWriteStruct("object", cv2.FILE_NODE_SEQ)
    for b, c in zip(boxes, cats):
        fs.startWriteStruct("", cv2.FILE_NODE_MAP)
        fs.write("name", c)
        fs.write("difficult", "0")
        fs.startWriteStruct("bndbox", cv2.FILE_NODE_MAP)
        fs.write("xmin", str(int(round(b[0])) + 1))
        fs.write("ymin", str(int(round(b[1])) + 1))
        fs.write("xmax", str(int(round(b[2]))))
        fs.write("ymax", str(int(round(b[3]))))
        fs.endWriteStruct()
        fs.endWriteStruct()
    fs.endWriteStruct()
    fs.endWriteStruct()
    fs.release()


def export(ds, prefix, out, classes, max_side=None):
    """max_side: shrink larger images (and their boxes) so the longer side is at most
    this many pixels, matching the pre-processing of our Python pipeline."""
    ids = []
    for name in ds.names:
        img_id = prefix + os.path.splitext(name)[0]
        src = os.path.join(ds.folder, name)
        img = cv2.imread(src)
        boxes = ds.ann[name]
        cats = ds.categories[name] if ds.categories else ["object"] * len(boxes)
        classes.update(cats)
        if max_side and max(img.shape[:2]) > max_side:
            f = max_side / max(img.shape[:2])
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            boxes = [[v * f for v in b] for b in boxes]
            cv2.imwrite(os.path.join(out, "JPEGImages", img_id + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            shutil.copy(src, os.path.join(out, "JPEGImages", img_id + ".jpg"))
        write_xml(os.path.join(out, "Annotations", img_id + ".xml"), img_id + ".jpg", img.shape, boxes, cats)
        write_yml(os.path.join(out, "Annotations", img_id + ".yml"), boxes, cats)
        ids.append(img_id)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--test", default="synthetic", choices=["synthetic", "real"])
    ap.add_argument("--max-side", type=int, help="shrink test images to this longer side")
    a = ap.parse_args()
    for d in ("JPEGImages", "Annotations", "ImageSets/Main"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    classes = set()
    train = export(JsonDataset(os.path.join(ROOT, "data", "synthetic", "train")), "tr_", a.out, classes)
    test_dir = os.path.join(ROOT, "data", "synthetic", "test") if a.test == "synthetic" else os.path.join(ROOT, "data", "real")
    test = export(JsonDataset(test_dir), "te_", a.out, classes, a.max_side)
    main_dir = os.path.join(a.out, "ImageSets", "Main")
    for fname, items in (("train.txt", train), ("test.txt", test), ("trainval.txt", train), ("class.txt", sorted(classes))):
        # CRLF: the authors' CmFile::loadStrList drops the last character of each line
        with open(os.path.join(main_dir, fname), "w", newline="") as f:
            f.write("".join(i + "\r\n" for i in items))
    print(f"exported {len(train)} train / {len(test)} test images to {a.out}")


if __name__ == "__main__":
    main()
