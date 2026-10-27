"""Colour space comparison on PASCAL VOC 2007: BING on RGB vs HSV vs Grayscale.

  python scripts/run_colorspace.py --voc-root /data/voc/VOCdevkit/VOC2007

The paper computes the normed gradient in RGB, HSV and Gray and merges the three ranked
lists ("BING-Diversified"). Here one BING model per colour space is trained on VOC 2007
trainval (final settings: released-code window sizes, greedy NMS) and each is evaluated on
the full test set: recall vs #proposals, box tightness, recall by object size and class,
and speed on one thread. The merged RGB + HSV + Gray list is evaluated as well.
Also scores the synthetic-trained RGB / HSV / Gray models on the real photographs.
Writes results/colorspace.json and models/bing_voc_{hsv,gray}_released_sizes_greedy.pkl.
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
from bing.datasets import ROOT, VOCDataset, get_dataset  # noqa: E402
from bing.model import RELEASED_SIZES  # noqa: E402
from measure_speed import fps, set_threads  # noqa: E402
from run_experiments import interleave  # noqa: E402
from run_voc import evaluate, load_test  # noqa: E402

SPACES = ("RGB", "HSV", "GRAY")


def model_path(cs):
    if cs == "RGB":  # the final model of section 5.1
        return os.path.join(ROOT, "models", "bing_voc_rgb_released_sizes_greedy.pkl")
    return os.path.join(ROOT, "models", f"bing_voc_{cs.lower()}_released_sizes_greedy.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc-root", required=True)
    a = ap.parse_args()
    out = os.path.join(ROOT, "results", "colorspace.json")
    R = {"voc": {}, "real": {}, "train_seconds": {}, "speed_1_thread": {}}
    models = {}
    for cs in SPACES:
        p = model_path(cs)
        if os.path.exists(p):
            models[cs] = BING.load(p)
        else:
            t0 = time.time()
            m = BING(sizes=RELEASED_SIZES, clip_large=True, interp="linear", nms="greedy", colorspace=cs)
            m.train(VOCDataset(a.voc_root, "trainval"))
            m.save(p)
            models[cs] = m
            R["train_seconds"][cs] = time.time() - t0
        print(f"[{cs}] model ready", flush=True)

    test = load_test(a.voc_root)
    img0 = cv2.imread(os.path.join(a.voc_root, "JPEGImages", test[0][0] + ".jpg"))
    for cs in SPACES:
        models[cs].propose(img0)  # JIT warm-up
        r = evaluate(lambda i, img, m=models[cs]: m.propose(img)[0], test, a.voc_root)
        R["voc"][cs] = r
        print(f"[{cs}] DR@100={r['dr_at'][100]:.3f} DR@1000={r['dr_at'][1000]:.3f} "
              f"MABO@1000={r['mabo_at'][1000]:.3f}", flush=True)
    r = evaluate(lambda i, img: interleave([models[c].propose(img)[0] for c in SPACES], 5000), test, a.voc_root)
    R["voc"]["RGB + HSV + GRAY"] = r
    print(f"[merged] DR@100={r['dr_at'][100]:.3f} DR@1000={r['dr_at'][1000]:.3f}", flush=True)

    # speed, one thread, the 300 VOC test images used elsewhere for speed
    ids = json.load(open(os.path.join(ROOT, "results", "voc.json")))["ss_subset_ids"]
    imgs = [cv2.imread(os.path.join(a.voc_root, "JPEGImages", i + ".jpg")) for i in ids]
    set_threads(1)
    for cs in SPACES:
        R["speed_1_thread"][cs] = max(fps(models[cs], imgs) for _ in range(2))
        print(f"[{cs}] {R['speed_1_thread'][cs]:.0f} img/s (1 thread)", flush=True)
    set_threads(4)

    # real photographs, models trained on the synthetic scenes (no real training images)
    real = list(get_dataset("real", "test"))
    for cs in SPACES:
        m = BING.load(os.path.join(ROOT, "models", f"bing_synthetic_{cs.lower()}.pkl"))
        ev = ProposalEvaluator(5000)
        for img, gt in real:
            ev.add(gt, m.propose(img)[0])
        R["real"][cs] = ev.summary()
        print(f"[{cs}] real photos DR@1000={R['real'][cs]['dr_at'][1000]:.3f}", flush=True)
    with open(out, "w") as f:
        json.dump(R, f, indent=1)
    print("saved", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
