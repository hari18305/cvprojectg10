"""Tier-1 experiments: full Selective Search run, per-category / per-size recall,
and a like-for-like comparison with the authors' original C++ BING.

  python scripts/run_tier1.py                                  # SS + breakdowns
  python scripts/run_tier1.py --orig-synth /tmp/voc_synth \
                              --orig-real /tmp/voc_real        # + original C++ results

--orig-* point at dataset folders produced by export_voc_format.py on which
baselines/original_bing/run_original.sh has already been run.
Writes results/tier1_synthetic.json.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bing import BING, ProposalEvaluator  # noqa: E402
from bing.datasets import ROOT, JsonDataset  # noqa: E402
from bing.metrics import best_overlap_curve  # noqa: E402
from run_experiments import selective_search  # noqa: E402

NS = (10, 100, 1000)


def size_bucket(b):
    s = np.sqrt((b[2] - b[0]) * (b[3] - b[1]))
    return "small (<32 px)" if s < 32 else ("medium (32-96 px)" if s < 96 else "large (>96 px)")


class Breakdown:
    """Per-object best overlaps at fixed budgets, grouped by category and by size."""

    def __init__(self):
        self.rows = []  # (category, size bucket, {N: best IoU})

    def add(self, gt, cats, props):
        B = best_overlap_curve(np.asarray(gt, float), props, max(NS))
        for g, c, bo in zip(gt, cats, B):
            self.rows.append((c, size_bucket(g), {n: float(bo[n - 1]) for n in NS}))

    def table(self, key):
        idx = 0 if key == "category" else 1
        out = {}
        for grp in sorted({r[idx] for r in self.rows}):
            rs = [r[2] for r in self.rows if r[idx] == grp]
            out[grp] = {"n": len(rs), **{f"dr@{n}": float(np.mean([r[n] >= 0.5 for r in rs])) for n in NS},
                        "mabo@1000": float(np.mean([r[1000] for r in rs]))}
        return out


def evaluate(fn, ds, with_breakdown=True):
    ev, bd = ProposalEvaluator(5000), Breakdown()
    for name in ds.names:
        img = cv2.imread(os.path.join(ds.folder, name))
        gt = np.array(ds.ann[name], float).reshape(-1, 4)
        t = time.perf_counter()
        props = fn(name, img)
        ev.add(gt, props, time.perf_counter() - t)
        if with_breakdown and len(gt):
            bd.add(gt, ds.categories[name], props)
    r = ev.summary()
    r["by_category"], r["by_size"] = bd.table("category"), bd.table("size")
    return r


def load_original(folder, prefix, name, orig_shape):
    """Read the C++ output: first line = count, then 'score, x1, y1, x2, y2' (1-based).
    Boxes are mapped back to the original resolution if the export shrank the image."""
    exported = cv2.imread(os.path.join(folder, "JPEGImages", prefix + os.path.splitext(name)[0] + ".jpg"))
    scale = orig_shape[1] / exported.shape[1]
    files = glob.glob(os.path.join(folder, "Results", "BBoxes*", prefix + os.path.splitext(name)[0] + ".txt"))
    if not files:
        raise FileNotFoundError(f"no C++ result for {name} in {folder}")
    rows = [list(map(float, l.split(","))) for l in open(files[0]).read().split("\n")[1:] if l.strip()]
    b = np.array([r[1:] for r in rows], float).reshape(-1, 4)
    b[:, :2] -= 1  # back to 0-based, end-exclusive
    return b * scale


def parse_seconds(folder):
    """Average prediction time printed by the C++ run (saved to run.log)."""
    log = os.path.join(folder, "run.log")
    if not os.path.exists(log):
        return None
    m = re.findall(r"Average time for predicting an image \(\w+\) is ([0-9.e-]+)s", open(log).read())
    return float(m[-1]) if m else None


def brief(tag, r):
    print(f"{tag:34s} DR@100={r['dr_at'][100]:.3f} DR@1000={r['dr_at'][1000]:.3f} "
          f"MABO@1000={r['mabo_at'][1000]:.3f} AUC={r['auc']:.3f} ({r['avg_proposals']:.0f} props/img)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-synth")
    ap.add_argument("--orig-real")
    ap.add_argument("--orig-real-500")
    ap.add_argument("--orig-synth-4t", help="same run as --orig-synth with 4 threads (timing only)")
    ap.add_argument("--skip-ss", action="store_true")
    a = ap.parse_args()
    out_path = os.path.join(ROOT, "results", "tier1_synthetic.json")
    R = json.load(open(out_path)) if os.path.exists(out_path) else {}

    test = JsonDataset(os.path.join(ROOT, "data", "synthetic", "test"))
    real = JsonDataset(os.path.join(ROOT, "data", "real"))
    model = BING.load(os.path.join(ROOT, "models", "bing_synthetic_rgb.pkl"))
    model.propose(np.zeros((64, 64, 3), np.uint8))  # JIT warm-up

    for split, ds in (("test", test), ("real", real)):
        R.setdefault(split, {})
        r = evaluate(lambda n, img: model.propose(img)[0], ds)
        R[split]["BING (ours)"] = r
        brief(f"[{split}] BING (ours)", r)
        if not a.skip_ss:
            r = evaluate(lambda n, img: selective_search(img), ds)
            R[split]["Selective Search (fast)"] = r
            brief(f"[{split}] Selective Search (all images)", r)
        runs = {"test": [("BING (authors' C++)", a.orig_synth)],
                "real": [("BING (authors' C++), original size", a.orig_real),
                         ("BING (authors' C++), shrunk to 500 px", a.orig_real_500)]}[split]
        for tag, orig in runs:
            if not orig:
                continue
            r = evaluate(lambda n, img: load_original(orig, "te_", n, img.shape), ds)
            r["sec_per_img_reported"] = parse_seconds(orig)
            R[split][tag] = r
            brief(f"[{split}] {tag}", r)

    if a.orig_synth_4t:
        R["orig_4thread_sec"] = parse_seconds(a.orig_synth_4t)
    with open(out_path, "w") as f:
        json.dump(R, f, indent=1)
    print("saved", out_path)


if __name__ == "__main__":
    main()
