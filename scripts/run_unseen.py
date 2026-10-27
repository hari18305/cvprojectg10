"""Generalisation to unseen classes, following the BING paper: train on 6 VOC classes,
test on the other 14.

  python scripts/run_unseen.py --voc-root /data/voc/VOCdevkit/VOC2007

The paper trains on bird, car, cat, cow, dog and sheep and tests on the remaining 14
classes, to check that "objectness" is generic rather than class-specific. We train BING
(final settings) on the VOC 2007 trainval images that contain a seen class, using only
the seen-class boxes as objects, and compare it on the test set with the model trained
on all 20 classes: recall on the 14 unseen classes, and on the 6 seen ones.
Writes results/unseen.json and models/bing_voc_rgb_6classes.pkl.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bing import BING, ProposalEvaluator  # noqa: E402
from bing.datasets import ROOT, VOCDataset  # noqa: E402
from bing.model import RELEASED_SIZES  # noqa: E402
from run_detection import load_split  # noqa: E402
from run_tier1 import Breakdown  # noqa: E402

SEEN = ("bird", "car", "cat", "cow", "dog", "sheep")


def evaluate(model, test, root, classes):
    ev, bd = ProposalEvaluator(5000), Breakdown()
    for i, (gt, names, diff) in test.items():
        keep = np.array([(n in classes) and not d for n, d in zip(names, diff)], bool)
        if not keep.any():
            continue
        img = cv2.imread(os.path.join(root, "JPEGImages", i + ".jpg"))
        b = model.propose(img)[0]
        g = gt[keep]
        ev.add(g, b)
        bd.add(g, [n for n, k in zip(names, keep) if k], b)
    r = ev.summary()
    r["by_class"] = bd.table("category")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc-root", required=True)
    a = ap.parse_args()
    t0 = time.time()
    train = VOCDataset(a.voc_root, "trainval", classes=SEEN)
    path = os.path.join(ROOT, "models", "bing_voc_rgb_6classes.pkl")
    m6 = BING(sizes=RELEASED_SIZES, clip_large=True, interp="linear", nms="greedy")
    m6.train(train)
    m6.save(path)
    m20 = BING.load(os.path.join(ROOT, "models", "bing_voc_rgb_released_sizes_greedy.pkl"))
    test = load_split(a.voc_root, "test")
    unseen = sorted({n for _, names, _ in test.values() for n in names} - set(SEEN))
    R = {"seen_classes": list(SEEN), "unseen_classes": unseen, "n_train_images_6": len(train),
         "train_seconds_6": time.time() - t0}
    for tag, model in (("trained on 6 classes", m6), ("trained on 20 classes", m20)):
        for part, cls in (("unseen 14 classes", unseen), ("seen 6 classes", SEEN)):
            r = evaluate(model, test, a.voc_root, set(cls))
            R[f"{tag} | {part}"] = r
            print(f"{tag:22s} {part:18s} objects={r['num_objects']:5d} DR@100={r['dr_at'][100]:.3f} "
                  f"DR@1000={r['dr_at'][1000]:.3f} DR@5000={r['dr_at'][5000]:.3f}", flush=True)
    with open(os.path.join(ROOT, "results", "unseen.json"), "w") as f:
        json.dump(R, f, indent=1)
    print("saved results/unseen.json")


if __name__ == "__main__":
    main()
