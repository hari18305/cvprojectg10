"""A complete detector on PASCAL VOC 2007: BING proposals + handcrafted features + linear SVMs.

  python scripts/run_detection.py --voc-root /data/voc/VOCdevkit/VOC2007 [--workers 4]

1. Proposals: BING (and BING + our multi-cue re-ranking) for every trainval and test image.
2. Training (trainval): positives are the ground-truth boxes and their mirror images;
   negatives are BING proposals overlapping no object (IoU < 0.3), then one round of
   hard-negative mining on 1500 trainval images.
3. Testing (test): the top N windows of each proposal method are scored by the 20 class
   SVMs, per-class NMS is applied and VOC 2007 average precision is computed.
   Random boxes and sliding windows are scored with the same classifier, for reference.

Writes models/detector_voc.pkl and results/detection_voc.json.
"""
import argparse
import json
import os
import pickle
import sys
import time
import xml.etree.ElementTree as ET
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bing import BING  # noqa: E402
from bing.datasets import ROOT  # noqa: E402
from bing.detector import VOC_CLASSES, Detector, average_precision, box_features  # noqa: E402
from bing.metrics import iou_matrix  # noqa: E402
from run_experiments import random_boxes, sliding_windows  # noqa: E402

BUDGETS = (100, 500, 1000)
SOURCES = ("BING", "BING + multi-cue re-ranking", "Random boxes", "Sliding windows")
G = {}  # per-worker state


def load_split(root, split):
    """{id: (boxes [M,4] 0-based, class names, difficult flags)}"""
    ids = [l.strip() for l in open(os.path.join(root, "ImageSets", "Main", split + ".txt")) if l.strip()]
    out = {}
    for i in ids:
        boxes, names, diff = [], [], []
        for obj in ET.parse(os.path.join(root, "Annotations", i + ".xml")).findall("object"):
            bb = obj.find("bndbox")
            x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            boxes.append([x1 - 1, y1 - 1, x2, y2])
            names.append(obj.findtext("name"))
            diff.append(bool(int(obj.findtext("difficult", "0"))))
        out[i] = (np.array(boxes, float).reshape(-1, 4), names, np.array(diff, bool))
    return out


def init_worker(root):
    cv2.setNumThreads(1)
    try:
        import numba
        numba.set_num_threads(1)
    except ImportError:
        pass
    G["root"] = root
    G["bing"] = BING.load(os.path.join(ROOT, "models", "bing_voc_rgb_released_sizes_greedy.pkl"))
    with open(os.path.join(ROOT, "models", "rerank_voc.pkl"), "rb") as f:
        G["rerank"] = pickle.load(f)


def imread(i):
    return cv2.imread(os.path.join(G["root"], "JPEGImages", i + ".jpg"))


def proposals(img, n=1000):
    b, s = G["bing"].propose(img)
    rb = G["rerank"].rerank(img, b, s)[0]
    return {"BING": b[:n], "BING + multi-cue re-ranking": rb[:n],
            "Random boxes": random_boxes(img, n), "Sliding windows": sliding_windows(img, n)}


# ------------------------------------------------------------------ training
def train_samples(args):
    """Ground-truth features (with mirrors) and random background proposals for one image."""
    i, (gt, names, diff), seed = args
    img = imread(i)
    keep = ~diff
    pos = box_features(img, gt[keep], flip=True)
    pos_cls = np.repeat([VOC_CLASSES.index(n) for n, k in zip(names, keep) if k], 2)
    b, _ = G["bing"].propose(img)
    b = b[:1000]
    bg = b[iou_matrix(gt, b).max(0) < 0.3] if len(gt) else b
    rng = np.random.default_rng(seed)
    bg = bg[rng.choice(len(bg), min(len(bg), 15), replace=False)]
    return pos, pos_cls, box_features(img, bg)


def hard_negatives(args):
    """BING proposals that the current SVMs score highly but that overlap no object of
    that class (IoU < 0.3): the hardest mistakes, per class."""
    i, (gt, names, diff), W, b0 = args
    img = imread(i)
    b, _ = G["bing"].propose(img)
    b = b[:500]
    f = box_features(img, b)
    s = f @ W.T + b0
    out = {}
    for j, c in enumerate(VOC_CLASSES):
        g = gt[[n == c for n in names]]
        ok = iou_matrix(g, b).max(0) < 0.3 if len(g) else np.ones(len(b), bool)
        idx = np.nonzero(ok & (s[:, j] > -1.0))[0]
        idx = idx[np.argsort(-s[idx, j])][:10]
        if len(idx):
            out[j] = f[idx]
    return out


# ------------------------------------------------------------------- testing
def test_image(args):
    """Detections of every proposal method and window budget for one test image."""
    k, i, det = args
    img = imread(i)
    out, secs = {}, {}
    for name, boxes in proposals(img).items():
        t = time.perf_counter()
        sc = det.scores(box_features(img, boxes))
        secs[name] = time.perf_counter() - t
        for n in BUDGETS:
            for j, d in det.detect(boxes[:n], sc[:n]).items():
                out[(name, n, j)] = np.c_[np.full(len(d), k, np.float32), d]
    return out, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc-root", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mine-images", type=int, default=1500)
    ap.add_argument("--limit", type=int, default=0, help="use only this many train/test images (quick check)")
    a = ap.parse_args()
    t0 = time.time()
    train = load_split(a.voc_root, "trainval")
    test = load_split(a.voc_root, "test")
    if a.limit:
        train = dict(list(train.items())[:a.limit])
        test = dict(list(test.items())[:a.limit])
    pool = Pool(a.workers, init_worker, (a.voc_root,))

    # 1. initial SVMs: ground truth vs. random background proposals
    pos, pos_cls, bg = [], [], []
    for p, pc, n in pool.imap(train_samples, [(i, v, k) for k, (i, v) in enumerate(train.items())], 16):
        pos.append(p), pos_cls.append(pc), bg.append(n)
    pos, pos_cls, bg = np.concatenate(pos), np.concatenate(pos_cls), np.concatenate(bg)
    print(f"[train] {len(pos)} positive patches (with mirrors), {len(bg)} background, "
          f"{time.time() - t0:.0f}s", flush=True)
    det = Detector()

    def fit_all(hard=None):
        Ws, bs = [], []
        for j, c in enumerate(VOC_CLASSES):
            neg = [bg, pos[pos_cls != j]] + ([hard[j]] if hard is not None and len(hard[j]) else [])
            w, b = det.fit_class(pos[pos_cls == j], np.concatenate(neg))
            Ws.append(w), bs.append(b)
        det.W, det.b = np.stack(Ws), np.array(bs, np.float32)
    fit_all()
    print(f"[train] initial SVMs, {time.time() - t0:.0f}s", flush=True)

    # 2. one round of hard-negative mining
    rng = np.random.default_rng(1)
    ids = sorted(rng.choice(list(train), min(a.mine_images, len(train)), replace=False))
    hard = {j: [] for j in range(len(VOC_CLASSES))}
    for out in pool.imap(hard_negatives, [(i, train[i], det.W, det.b) for i in ids], 8):
        for j, f in out.items():
            hard[j].append(f)
    hard = {j: (np.concatenate(v)[:20000] if v else np.zeros((0, pos.shape[1]), np.float32)) for j, v in hard.items()}
    print(f"[train] mined {sum(len(v) for v in hard.values())} hard negatives, {time.time() - t0:.0f}s", flush=True)
    fit_all(hard)
    tag = f"_limit{a.limit}" if a.limit else ""
    with open(os.path.join(ROOT, "models", f"detector_voc{tag}.pkl"), "wb") as f:
        pickle.dump(det, f)
    train_s = time.time() - t0
    print(f"[train] done in {train_s:.0f}s", flush=True)

    # 3. test: score the top windows of every method, then per-class NMS + VOC AP
    dets = {}
    feat_time = {s: 0.0 for s in SOURCES}
    test_ids = list(test)
    jobs = [(k, i, det) for k, i in enumerate(test_ids)]
    for k, (out, secs) in enumerate(pool.imap(test_image, jobs, 4)):
        for key, d in out.items():
            dets.setdefault(key, []).append(d)
        for s, dt in secs.items():
            feat_time[s] += dt
        if (k + 1) % 500 == 0:
            print(f"[test] {k + 1}/{len(test)} images, {time.time() - t0:.0f}s", flush=True)
    pool.close()

    R = {"n_train": len(train), "n_test": len(test), "train_seconds": train_s,
         "features": "HOG (64x64, 9 bins, 1764-D) + 8x4 hue/saturation histogram",
         "n_hard_negatives": {c: len(hard[j]) for j, c in enumerate(VOC_CLASSES)}, "results": {}}
    gts = {j: {k: (test[i][0][[x == c for x in test[i][1]]], test[i][2][[x == c for x in test[i][1]]])
               for k, i in enumerate(test_ids)} for j, c in enumerate(VOC_CLASSES)}
    for s in SOURCES:
        for n in BUDGETS:
            aps = {}
            for j, c in enumerate(VOC_CLASSES):
                d = dets.get((s, n, j), [])
                d = np.concatenate(d) if d else np.zeros((0, 6), np.float32)
                aps[c] = average_precision(d, gts[j])
            m = float(np.mean(list(aps.values())))
            R["results"][f"{s} @{n}"] = {"source": s, "windows": n, "mAP": m, "AP": aps,
                                         "classify_ms_per_img@1000": feat_time[s] / len(test) * 1e3}
            print(f"{s:30s} top {n:4d} windows: mAP {100 * m:.1f}%", flush=True)
    H = W = 500
    R["sliding_windows_per_500px_image"] = int(len(sliding_windows(np.zeros((H, W, 3), np.uint8), 10 ** 9)))
    with open(os.path.join(ROOT, "results", f"detection_voc{tag}.json"), "w") as f:
        json.dump(R, f, indent=1)
    print(f"saved results/detection_voc.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
