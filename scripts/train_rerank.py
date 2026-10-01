"""Train the multi-cue re-ranker (bing/rerank.py) on PASCAL VOC 2007 trainval proposals.

  python scripts/train_rerank.py --voc-root /data/voc/VOCdevkit/VOC2007

Writes models/rerank_voc.pkl.
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bing import BING  # noqa: E402
from bing.datasets import ROOT, VOCDataset  # noqa: E402
from bing.metrics import iou_matrix  # noqa: E402
from bing.rerank import Reranker, features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc-root", required=True)
    ap.add_argument("--images", type=int, default=2500, help="trainval images used")
    ap.add_argument("--neg-per-img", type=int, default=300)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    model = BING.load(os.path.join(ROOT, "models", "bing_voc_rgb_released_sizes_greedy.pkl"))
    X, y = [], []
    t0 = time.time()
    for k, (img, gt) in enumerate(VOCDataset(a.voc_root, "trainval", a.images)):
        if not len(gt):
            continue
        b, s = model.propose(img)
        lab = iou_matrix(gt, b).max(0) >= 0.5
        f = features(img, b, s)
        pos = np.nonzero(lab)[0]
        neg = np.nonzero(~lab)[0]
        neg = rng.choice(neg, min(len(neg), a.neg_per_img), replace=False)
        X.append(f[np.concatenate([pos, neg])])
        y.append(np.r_[np.ones(len(pos)), np.zeros(len(neg))])
        if (k + 1) % 500 == 0:
            print(f"{k + 1} images, {time.time() - t0:.0f}s", flush=True)
    X, y = np.concatenate(X), np.concatenate(y)
    print(f"training on {len(y)} proposals ({int(y.sum())} positive)", flush=True)
    r = Reranker().fit(X, y)
    with open(os.path.join(ROOT, "models", "rerank_voc.pkl"), "wb") as f:
        pickle.dump(r, f)
    print(f"saved models/rerank_voc.pkl ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
