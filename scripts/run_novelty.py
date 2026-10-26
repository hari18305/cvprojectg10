"""Our extensions of BING evaluated on the full PASCAL VOC 2007 test set.

  * multi-cue re-ranking (bing/rerank.py, trained by scripts/train_rerank.py):
    colour contrast, edge density and geometry cues re-order BING's proposals.
    This is the extension that works (better recall at every budget, +14 points
    on small objects).
  * two extensions that were tried and did NOT help, kept for the record:
    fine-scale 8 px windows (models/bing_voc_rgb_fine_scale.pkl) and
    gradient-guided box refinement (bing/refine.py).

  python scripts/run_novelty.py --voc-root /data/voc/VOCdevkit/VOC2007

Writes results/novelty_voc.json.
"""
import argparse
import json
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bing import BING  # noqa: E402
from bing.datasets import ROOT  # noqa: E402
from run_voc import evaluate, load_test  # noqa: E402

REFINE = {"frac": 0.1, "min_gain": 1.0, "iters": 1, "top": 1000}
CONFIGS = [  # name, BING model, refinement settings, use the re-ranker
    ("BING (final baseline)", "bing_voc_rgb_released_sizes_greedy.pkl", None, False),
    ("+ edge refinement", "bing_voc_rgb_released_sizes_greedy.pkl", REFINE, False),
    ("+ fine-scale windows", "bing_voc_rgb_fine_scale.pkl", None, False),
    ("BING+ (both)", "bing_voc_rgb_fine_scale.pkl", REFINE, False),
    ("BING + multi-cue re-ranking (ours)", "bing_voc_rgb_released_sizes_greedy.pkl", None, True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc-root", required=True)
    a = ap.parse_args()
    test = load_test(a.voc_root)
    R = {"refine": REFINE}
    with open(os.path.join(ROOT, "models", "rerank_voc.pkl"), "rb") as f:
        reranker = pickle.load(f)
    for name, model_file, refine, rerank in CONFIGS:
        m = BING.load(os.path.join(ROOT, "models", model_file), refine=refine)

        def fn(i, img, m=m, rerank=rerank):
            b, s = m.propose(img)
            return reranker.rerank(img, b, s)[0] if rerank else b
        r = evaluate(fn, test, a.voc_root)
        R[name] = r
        print(f"{name:24s} DR@100={r['dr_at'][100]:.3f} DR@1000={r['dr_at'][1000]:.3f} "
              f"DR@5000={r['dr_at'][5000]:.3f} recall@IoU0.7={r['recall_iou']['0.70']:.3f} "
              f"MABO={r['mabo_at'][1000]:.3f} small={r['by_size']['small (<32 px)']['dr@1000']:.3f} "
              f"{r['fps']:.0f} img/s", flush=True)
    with open(os.path.join(ROOT, "results", "novelty_voc.json"), "w") as f:
        json.dump(R, f, indent=1)
    print("saved results/novelty_voc.json")


if __name__ == "__main__":
    main()
