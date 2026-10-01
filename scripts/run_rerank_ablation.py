"""Which cues does the multi-cue re-ranker rely on? Retrain it without each cue group
and evaluate on the full VOC 2007 test set. Adds entries to results/novelty_voc.json.

  python scripts/run_rerank_ablation.py --voc-root /data/voc/VOCdevkit/VOC2007
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bing import BING  # noqa: E402
from bing.datasets import ROOT, VOCDataset  # noqa: E402
from bing.metrics import iou_matrix  # noqa: E402
from bing.rerank import Reranker, features  # noqa: E402
from run_voc import evaluate, load_test  # noqa: E402

GROUPS = {"colour contrast": [2, 3], "edge density": [4, 5, 6], "geometry": list(range(7, 16))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc-root", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    m = BING.load(os.path.join(ROOT, "models", "bing_voc_rgb_released_sizes_greedy.pkl"))
    X, y = [], []
    for img, gt in VOCDataset(a.voc_root, "trainval", 2500):
        if not len(gt):
            continue
        b, s = m.propose(img)
        lab = iou_matrix(gt, b).max(0) >= 0.5
        f = features(img, b, s)
        pos, neg = np.nonzero(lab)[0], np.nonzero(~lab)[0]
        neg = rng.choice(neg, min(len(neg), 300), replace=False)
        X.append(f[np.concatenate([pos, neg])])
        y.append(np.r_[np.ones(len(pos)), np.zeros(len(neg))])
    X, y = np.concatenate(X), np.concatenate(y)
    test = load_test(a.voc_root)
    path = os.path.join(ROOT, "results", "novelty_voc.json")
    R = json.load(open(path))
    for name, cols in GROUPS.items():
        keep = [c for c in range(X.shape[1]) if c not in cols]
        r = Reranker().fit(X[:, keep], y)

        def fn(i, img, r=r, keep=keep):
            b, s = m.propose(img)
            p = r.clf.predict_proba(features(img, b, s)[:, keep])[:, 1]
            return b[np.argsort(-p, kind="stable")]
        res = evaluate(fn, test, a.voc_root)
        R[f"re-ranking without {name}"] = res
        print(f"without {name:16s} DR@100={res['dr_at'][100]:.3f} DR@1000={res['dr_at'][1000]:.3f} "
              f"small={res['by_size']['small (<32 px)']['dr@1000']:.3f}", flush=True)
        json.dump(R, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
