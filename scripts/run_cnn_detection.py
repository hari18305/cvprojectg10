"""Detector on BING proposals with CNN features (frozen ResNet-50) instead of HOG.

  python scripts/run_cnn_detection.py --voc-root /data/voc/VOCdevkit/VOC2007

Same protocol as scripts/run_detection.py (train on VOC 2007 trainval, test on test,
VOC 11-point AP), with:
  * features: one ResNet-50 pass per image (ImageNet weights, no fine-tuning), each window
    pooled as the mean conv4 + conv5 activations inside it (bing/cnn.py);
  * 20 one-vs-rest linear SVMs, trained on ground truth + windows with IoU >= 0.7 vs.
    background windows, then one round of hard-negative mining;
  * class-specific box regression (R-CNN), which tightens BING's coarse boxes.
Needs torch + torchvision. Writes models/cnn_detector_voc.pkl and results/cnn_detection_voc.json.
"""
import argparse
import json
import os
import pickle
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bing import BING  # noqa: E402
from bing.cnn import CNNDetector, CNNFeatures, box_targets, fit_ridge, train_linear  # noqa: E402
from bing.datasets import ROOT  # noqa: E402
from bing.detector import VOC_CLASSES, average_precision  # noqa: E402
from bing.metrics import iou_matrix  # noqa: E402
from run_detection import load_split  # noqa: E402
from run_experiments import random_boxes, sliding_windows  # noqa: E402

BUDGETS = (100, 300, 1000)
SOURCES = ("BING", "BING + multi-cue re-ranking", "Random boxes", "Sliding windows")
K = len(VOC_CLASSES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc-root", required=True)
    ap.add_argument("--mine-images", type=int, default=1500)
    ap.add_argument("--limit", type=int, default=0, help="use only this many train/test images (quick check)")
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(0)
    train, test = load_split(a.voc_root, "trainval"), load_split(a.voc_root, "test")
    if a.limit:
        train, test = dict(list(train.items())[:a.limit]), dict(list(test.items())[:a.limit])
    bing = BING.load(os.path.join(ROOT, "models", "bing_voc_rgb_released_sizes_greedy.pkl"))
    with open(os.path.join(ROOT, "models", "rerank_voc.pkl"), "rb") as f:
        reranker = pickle.load(f)
    cnn = CNNFeatures()
    imread = lambda i: cv2.imread(os.path.join(a.voc_root, "JPEGImages", i + ".jpg"))

    # 1. training samples: ground truth (+ windows with IoU >= 0.7) vs. background windows,
    #    and windows with IoU >= 0.6 for the box regressors
    X, Y, M = [], [], []
    regX, regT, regC = [], [], []
    for k, (i, (gt, names, diff)) in enumerate(train.items()):
        img = imread(i)
        maps = cnn.maps(img)
        b, _ = bing.propose(img)
        b = b[:1000]
        cls = np.array([VOC_CLASSES.index(n) for n in names], int)
        keep = ~diff
        g, gc = gt[keep], cls[keep]
        iou = iou_matrix(gt, b) if len(gt) else np.zeros((0, len(b)))
        # positives: the ground-truth boxes and up to 3 close windows per object
        pos_b, pos_c = [g], [gc]
        reg_b, reg_t, reg_c = [], [], []
        for o in np.nonzero(keep)[0]:
            near = np.nonzero(iou[o] >= 0.7)[0][:3]
            pos_b.append(b[near]), pos_c.append(np.full(len(near), cls[o]))
            r = np.nonzero(iou[o] >= 0.6)[0][:10]
            if len(r):
                reg_b.append(b[r]), reg_t.append(box_targets(b[r], np.repeat(gt[o:o + 1], len(r), 0)))
                reg_c.append(np.full(len(r), cls[o]))
        pb, pc = np.concatenate(pos_b), np.concatenate(pos_c)
        bg = b[iou.max(0) < 0.3] if len(gt) else b
        bg = bg[rng.choice(len(bg), min(len(bg), 15), replace=False)]
        f = cnn.pool(maps, np.concatenate([pb, bg]))
        y = -np.ones((len(f), K), np.int8)
        y[np.arange(len(pb)), pc] = 1
        X.append(f.astype(np.float16)), Y.append(y), M.append(np.ones((len(f), K), np.uint8))
        if reg_b:
            rb = np.concatenate(reg_b)
            regX.append(cnn.pool(maps, rb).astype(np.float16)), regT.append(np.concatenate(reg_t))
            regC.append(np.concatenate(reg_c))
        if (k + 1) % 500 == 0:
            print(f"[train] {k + 1}/{len(train)} images, {time.time() - t0:.0f}s", flush=True)
    X, Y, M = np.concatenate(X), np.concatenate(Y), np.concatenate(M)
    print(f"[train] {int((Y > 0).any(1).sum())} positive and {int((Y < 0).all(1).sum())} background windows, "
          f"{time.time() - t0:.0f}s", flush=True)
    W, bb = train_linear(X.astype(np.float32), Y, M)
    print(f"[train] initial classifiers, {time.time() - t0:.0f}s", flush=True)

    # 2. one round of hard-negative mining: background windows the classifiers score highest
    ids = sorted(rng.choice(list(train), min(a.mine_images, len(train)), replace=False))
    HX, HM = [], []
    for i in ids:
        gt, names, diff = train[i]
        img = imread(i)
        b, _ = bing.propose(img)
        b = b[:500]
        f = cnn.pool(cnn.maps(img), b)
        s = f @ W.T + bb
        mask = np.zeros((len(b), K), np.uint8)
        for j, c in enumerate(VOC_CLASSES):
            g = gt[[n == c for n in names]]
            ok = iou_matrix(g, b).max(0) < 0.3 if len(g) else np.ones(len(b), bool)
            idx = np.nonzero(ok & (s[:, j] > -1.0))[0]
            mask[idx[np.argsort(-s[idx, j])][:10], j] = 1
        sel = mask.any(1)
        HX.append(f[sel].astype(np.float16)), HM.append(mask[sel])
    HX, HM = np.concatenate(HX), np.concatenate(HM)
    print(f"[train] mined {int(HM.sum())} hard negatives, {time.time() - t0:.0f}s", flush=True)
    X2 = np.concatenate([X, HX]).astype(np.float32)
    Y2 = np.concatenate([Y, -np.ones(HM.shape, np.int8)])
    M2 = np.concatenate([M, HM])
    W, bb = train_linear(X2, Y2, M2)
    del X2

    # 3. box regressors, one per class
    regX, regT, regC = np.concatenate(regX).astype(np.float64), np.concatenate(regT), np.concatenate(regC)
    R = np.zeros((K, regX.shape[1], 4), np.float32)
    r0 = np.zeros((K, 4), np.float32)
    for j in range(K):
        sel = regC == j
        if sel.sum() > 20:
            R[j], r0[j] = fit_ridge(regX[sel], regT[sel])
    det = CNNDetector(W.astype(np.float32), bb.astype(np.float32), R, r0)
    tag = f"_limit{a.limit}" if a.limit else ""
    with open(os.path.join(ROOT, "models", f"cnn_detector_voc{tag}.pkl"), "wb") as f:
        pickle.dump(det, f)
    train_s = time.time() - t0
    print(f"[train] done in {train_s:.0f}s ({len(regC)} regression samples)", flush=True)

    # 4. test: score the top windows of every source, with and without box regression
    dets = {}
    secs = {"backbone": 0.0, "bing": 0.0, "rerank": 0.0}
    test_ids = list(test)
    for k, i in enumerate(test_ids):
        img = imread(i)
        t = time.perf_counter(); maps = cnn.maps(img); secs["backbone"] += time.perf_counter() - t
        t = time.perf_counter(); b, s = bing.propose(img); secs["bing"] += time.perf_counter() - t
        t = time.perf_counter(); rb = reranker.rerank(img, b, s)[0]; secs["rerank"] += time.perf_counter() - t
        srcs = {"BING": b[:1000], "BING + multi-cue re-ranking": rb[:1000],
                "Random boxes": random_boxes(img, 1000), "Sliding windows": sliding_windows(img, 1000)}
        for name, boxes in srcs.items():
            f = cnn.pool(maps, boxes)
            for n in BUDGETS:
                for reg in (False, True):
                    for j, d in det.detect(boxes[:n], f[:n], regress=reg).items():
                        dets.setdefault((name, n, reg, j), []).append(np.c_[np.full(len(d), k, np.float32), d])
        if (k + 1) % 500 == 0:
            print(f"[test] {k + 1}/{len(test)} images, {time.time() - t0:.0f}s", flush=True)

    gts = {j: {k: (test[i][0][[x == c for x in test[i][1]]], test[i][2][[x == c for x in test[i][1]]])
               for k, i in enumerate(test_ids)} for j, c in enumerate(VOC_CLASSES)}
    R_ = {"n_train": len(train), "n_test": len(test), "train_seconds": train_s,
          "features": "ResNet-50 (ImageNet, frozen), mean conv4 + conv5 activations per box, 3072-D",
          "ms_per_image": {k: v / len(test) * 1e3 for k, v in secs.items()}, "results": {}}
    for s_ in SOURCES:
        for n in BUDGETS:
            for reg in (False, True):
                aps = {}
                for j, c in enumerate(VOC_CLASSES):
                    d = dets.get((s_, n, reg, j), [])
                    aps[c] = average_precision(np.concatenate(d) if d else np.zeros((0, 6), np.float32), gts[j])
                m = float(np.nanmean(list(aps.values())))  # nan: class absent (only with --limit)
                key = f"{s_} @{n}" + (" +box regression" if reg else "")
                R_["results"][key] = {"source": s_, "windows": n, "box_regression": reg, "mAP": m, "AP": aps}
                print(f"{key:55s} mAP {100 * m:.1f}%", flush=True)
    with open(os.path.join(ROOT, "results", f"cnn_detection_voc{tag}.json"), "w") as f:
        json.dump(R_, f, indent=1)
    print(f"saved results/cnn_detection_voc{tag}.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
