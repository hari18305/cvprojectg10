"""Download PASCAL VOC 2007 and prepare it for both implementations.

  python scripts/prepare_voc.py --dir /path/to/voc

Downloads VOCtrainval / VOCtest (HTTPS mirror of the official files), extracts them
to <dir>/VOCdevkit/VOC2007, and writes the extra files the authors' C++ code needs:
a separate <dir>/VOC2007_original_bing folder (linked images, OpenCV YAML annotations,
CRLF ImageSets lists where train = trainval as in the paper). The VOC folder itself is
left untouched, so scripts/run_experiments.py --dataset voc reads it directly.
"""
import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_voc_format import write_yml  # noqa: E402

MIRROR = "https://data.pjreddie.com/files/"
FILES = ("VOCtrainval_06-Nov-2007.tar", "VOCtest_06-Nov-2007.tar")
CLASSES = ("aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse "
           "motorbike person pottedplant sheep sofa train tvmonitor").split()


def download(d):
    for f in FILES:
        path = os.path.join(d, f)
        if not os.path.exists(path):
            subprocess.run(["curl", "-sSfL", "--retry", "4", "-o", path + ".part", MIRROR + f], check=True)
            os.rename(path + ".part", path)
        if not os.path.exists(os.path.join(d, "VOCdevkit", "VOC2007", "ImageSets", "Main",
                                           "test.txt" if "test" in f else "trainval.txt")):
            subprocess.run(["tar", "-xf", path, "-C", d], check=True)


def prepare_for_original(root, out):
    """Folder for the authors' code: links to VOC's images and annotations, plus
    OpenCV YAML annotations (difficult objects kept and flagged, which their loader skips)
    and its own ImageSets/Main lists. The VOC folder itself is left untouched."""
    main = os.path.join(root, "ImageSets", "Main")
    os.makedirs(os.path.join(out, "ImageSets", "Main"), exist_ok=True)
    os.makedirs(os.path.join(out, "Annotations"), exist_ok=True)
    if not os.path.exists(os.path.join(out, "JPEGImages")):
        os.symlink(os.path.join(root, "JPEGImages"), os.path.join(out, "JPEGImages"))
    ids = {s: [l.strip() for l in open(os.path.join(main, s + ".txt")) if l.strip()] for s in ("trainval", "test")}
    for i in ids["trainval"] + ids["test"]:
        yml = os.path.join(out, "Annotations", i + ".yml")
        if os.path.exists(yml):
            continue
        tree = ET.parse(os.path.join(root, "Annotations", i + ".xml"))
        boxes, cats, diff = [], [], []
        for obj in tree.findall("object"):
            bb = obj.find("bndbox")
            # write_yml expects 0-based x1,y1 and adds 1 back, so undo VOC's 1-based origin here
            x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            boxes.append([x1 - 1, y1 - 1, x2, y2])
            cats.append(obj.findtext("name"))
            diff.append(obj.findtext("difficult", "0"))
        write_yml(yml, boxes, cats, diff)
    # CRLF: the authors' CmFile::loadStrList drops the last character of each line.
    # Their train.txt is VOC's trainval (the paper trains on trainval, tests on test).
    for name, items in (("train.txt", ids["trainval"]), ("test.txt", ids["test"]), ("class.txt", CLASSES)):
        with open(os.path.join(out, "ImageSets", "Main", name), "w", newline="") as f:
            f.write("".join(x + "\r\n" for x in items))
    print(f"prepared {len(ids['trainval'])} trainval / {len(ids['test'])} test images; authors' code folder: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    download(a.dir)
    prepare_for_original(os.path.join(a.dir, "VOCdevkit", "VOC2007"), os.path.join(a.dir, "VOC2007_original_bing"))


if __name__ == "__main__":
    main()
