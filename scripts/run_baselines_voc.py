"""Selective Search (all 4,952 VOC 2007 test images) and EdgeBoxes, scored with our evaluator.

  python scripts/run_baselines_voc.py --voc-root /data/voc/VOCdevkit/VOC2007 [--workers 4]

Selective Search: OpenCV's "fast" mode, as before (earlier only 300 random test images).
EdgeBoxes (Zitnick & Dollar, ECCV 2014): OpenCV's implementation on structured-forest
edges, with OpenCV's pre-trained edge model (downloaded once into third_party/), default
parameters (alpha 0.65, beta 0.75), up to 5000 boxes.
Both run one image per process, one thread each; seconds per image are single-thread.
Adds entries to results/voc.json.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bing.datasets import ROOT  # noqa: E402
from run_voc import brief, evaluate, load_test  # noqa: E402

EDGE_MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/cv/ximgproc/model.yml.gz"
EDGE_MODEL = os.path.join(ROOT, "third_party", "opencv_extra", "model.yml.gz")
G = {}


def init(root):
    cv2.setNumThreads(1)
    G["root"] = root
    G["edge"] = cv2.ximgproc.createStructuredEdgeDetection(EDGE_MODEL)


def selective_search(img, n=5000):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    r = ss.process()[:n].astype(float)
    return np.stack([r[:, 0], r[:, 1], r[:, 0] + r[:, 2], r[:, 1] + r[:, 3]], 1)


def edge_boxes(img, n=5000):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    e = G["edge"].detectEdges(rgb)
    o = G["edge"].computeOrientation(e)
    e = G["edge"].edgesNms(e, o)
    eb = cv2.ximgproc.createEdgeBoxes()
    eb.setMaxBoxes(n)
    r = eb.getBoundingBoxes(e, o)
    r = (r[0] if isinstance(r, tuple) else r).reshape(-1, 4).astype(float)
    return np.stack([r[:, 0], r[:, 1], r[:, 0] + r[:, 2], r[:, 1] + r[:, 3]], 1)


def run_one(i):
    img = cv2.imread(os.path.join(G["root"], "JPEGImages", i + ".jpg"))
    out = {}
    for name, fn in (("ss", selective_search), ("eb", edge_boxes)):
        t = time.perf_counter()
        b = fn(img)
        out[name] = (b.astype(np.float32), time.perf_counter() - t)
    return i, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc-root", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if not os.path.exists(EDGE_MODEL):
        os.makedirs(os.path.dirname(EDGE_MODEL), exist_ok=True)
        urllib.request.urlretrieve(EDGE_MODEL_URL, EDGE_MODEL)
    test = load_test(a.voc_root)
    props, secs = {"ss": {}, "eb": {}}, {"ss": [], "eb": []}
    t0 = time.time()
    with Pool(a.workers, init, (a.voc_root,)) as pool:
        for k, (i, out) in enumerate(pool.imap_unordered(run_one, [t[0] for t in test], 4)):
            for name, (b, s) in out.items():
                props[name][i] = b
                secs[name].append(s)
            if (k + 1) % 500 == 0:
                print(f"{k + 1}/{len(test)} images, {time.time() - t0:.0f}s", flush=True)
    path = os.path.join(ROOT, "results", "voc.json")
    R = json.load(open(path))
    for name, key in (("ss", "Selective Search (fast), all test images"), ("eb", "EdgeBoxes")):
        r = evaluate(lambda i, img, p=props[name]: p[i], test, a.voc_root, timed=False)
        r["sec_per_img_1_thread"] = float(np.mean(secs[name]))
        r["avg_proposals"] = float(np.mean([len(v) for v in props[name].values()]))
        R[key] = r
        brief(key, r)
        print(f"   {r['sec_per_img_1_thread']:.2f} s/img on one thread", flush=True)
    with open(path, "w") as f:
        json.dump(R, f, indent=1)
    print("saved results/voc.json")


if __name__ == "__main__":
    main()
