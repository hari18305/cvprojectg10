"""BING on PASCAL VOC 2007 with the paper's protocol (train on trainval, test on test).

  python scripts/prepare_voc.py --dir /data/voc                       # download + prepare
  bash baselines/original_bing/run_original.sh /data/voc/VOC2007_original_bing/ 1
  python scripts/run_voc.py --voc-root /data/voc/VOCdevkit/VOC2007 \
                            --orig /data/voc/VOC2007_original_bing

Evaluates our BING, the authors' C++ BING (if --orig is given), Selective Search on a
random test subset, and random boxes; reports DR / MABO curves plus recall per VOC class
and per object size. 'difficult' objects are ignored, as in the paper.
Writes results/voc.json and models/bing_voc_rgb.pkl.
"""
import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bing import BING, ProposalEvaluator  # noqa: E402
from bing.datasets import ROOT, VOCDataset  # noqa: E402
from bing.model import BASE_SIZES, RELEASED_SIZES  # noqa: E402
from run_experiments import random_boxes, selective_search  # noqa: E402
from run_tier1 import Breakdown  # noqa: E402


def load_test(root):
    """[(id, boxes [N,4] 0-based, class names)] for non-difficult objects."""
    ids = [l.strip() for l in open(os.path.join(root, "ImageSets", "Main", "test.txt")) if l.strip()]
    out = []
    for i in ids:
        tree = ET.parse(os.path.join(root, "Annotations", i + ".xml"))
        boxes, names = [], []
        for obj in tree.findall("object"):
            if int(obj.findtext("difficult", "0")):
                continue
            bb = obj.find("bndbox")
            x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            boxes.append([x1 - 1, y1 - 1, x2, y2])
            names.append(obj.findtext("name"))
        out.append((i, np.array(boxes, float).reshape(-1, 4), names))
    return out


def load_original(orig, img_id):
    path = None
    res = os.path.join(orig, "Results")
    for d in os.listdir(res):
        if d.startswith("BBoxes"):
            path = os.path.join(res, d, img_id + ".txt")
    rows = [list(map(float, l.split(","))) for l in open(path).read().split("\n")[1:] if l.strip()]
    b = np.array([r[1:] for r in rows], float).reshape(-1, 4)
    b[:, :2] -= 1
    return b


def evaluate(fn, test, root, timed=True):
    ev, bd = ProposalEvaluator(5000), Breakdown()
    for img_id, gt, names in test:
        img = cv2.imread(os.path.join(root, "JPEGImages", img_id + ".jpg"))
        t = time.perf_counter()
        props = fn(img_id, img)
        ev.add(gt, props, time.perf_counter() - t if timed else None)
        if len(gt):
            bd.add(gt, names, props)
    r = ev.summary()
    r["by_class"], r["by_size"] = bd.table("category"), bd.table("size")
    return r


def brief(tag, r):
    print(f"{tag:28s} DR@100={r['dr_at'][100]:.3f} DR@1000={r['dr_at'][1000]:.3f} "
          f"DR@5000={r['dr_at'][5000]:.3f} MABO@1000={r['mabo_at'][1000]:.3f} "
          f"({r['avg_proposals']:.0f} props/img, {r.get('fps', 0):.1f} img/s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc-root", required=True)
    ap.add_argument("--orig", help="authors' code folder, after run_original.sh")
    ap.add_argument("--ss-images", type=int, default=300)
    ap.add_argument("--sizes", choices=["paper", "released"], default="paper",
                    help="paper: {10..320}, windows larger than the image skipped; "
                         "released: {16..512} clipped to the image, as in the authors' code")
    ap.add_argument("--nms", choices=["local_max", "greedy"], default="local_max",
                    help="greedy = the authors' non-maximum suppression")
    ap.add_argument("--only-ours", action="store_true", help="skip baselines, add to results/voc.json")
    a = ap.parse_args()
    out = os.path.join(ROOT, "results", "voc.json")
    R = json.load(open(out)) if (a.only_ours and os.path.exists(out)) else {}
    extras = [x for x, on in (("released-code sizes", a.sizes == "released"), ("greedy NMS", a.nms == "greedy")) if on]
    tag = "BING (ours" + "".join(", " + x for x in extras) + ")"
    suffix = ("_released_sizes" if a.sizes == "released" else "") + ("_greedy" if a.nms == "greedy" else "")
    model_path = os.path.join(ROOT, "models", f"bing_voc_rgb{suffix}.pkl")

    t0 = time.time()
    if a.sizes == "paper":
        model = BING(interp="linear", nms=a.nms)
    else:
        model = BING(sizes=RELEASED_SIZES, clip_large=True, interp="linear", nms=a.nms)
    model.train(VOCDataset(a.voc_root, "trainval"))
    R.setdefault("train_seconds", {})
    if not isinstance(R["train_seconds"], dict):
        R["train_seconds"] = {"BING (ours)": R["train_seconds"]}
    R["train_seconds"][tag] = time.time() - t0
    R["n_train"] = len(VOCDataset(a.voc_root, "trainval"))
    model.save(model_path)

    test = load_test(a.voc_root)
    R["n_test"], R["n_objects"] = len(test), int(sum(len(g) for _, g, _ in test))
    print(f"test: {R['n_test']} images, {R['n_objects']} objects", flush=True)
    model.propose(cv2.imread(os.path.join(a.voc_root, "JPEGImages", test[0][0] + ".jpg")))  # JIT warm-up

    R[tag] = r = evaluate(lambda i, img: model.propose(img)[0], test, a.voc_root)
    brief(tag, r)
    if a.only_ours:
        with open(out, "w") as f:
            json.dump(R, f, indent=1)
        print("saved results/voc.json")
        return
    if a.orig:
        r = evaluate(lambda i, img: load_original(a.orig, i), test, a.voc_root, timed=False)
        log = os.path.join(a.orig, "run.log")
        if os.path.exists(log):
            import re
            m = re.findall(r"is ([0-9.e-]+)s", open(log).read())
            r["sec_per_img_reported"] = float(m[-1]) if m else None
        R["BING (authors' C++)"] = r
        brief("BING (authors' C++)", r)
    R["Random boxes"] = r = evaluate(lambda i, img: random_boxes(img), test, a.voc_root)
    brief("Random boxes", r)

    rng = np.random.default_rng(0)
    sub = [test[k] for k in sorted(rng.choice(len(test), min(a.ss_images, len(test)), replace=False))]
    R["ss_subset_ids"] = [s[0] for s in sub]
    R["Selective Search (fast)"] = r = evaluate(lambda i, img: selective_search(img), sub, a.voc_root)
    brief(f"Selective Search ({len(sub)} imgs)", r)
    R["BING (ours) @SS subset"] = r = evaluate(lambda i, img: model.propose(img)[0], sub, a.voc_root)
    brief("BING (ours) same subset", r)

    with open(out, "w") as f:
        json.dump(R, f, indent=1)
    print("saved results/voc.json")


if __name__ == "__main__":
    main()
