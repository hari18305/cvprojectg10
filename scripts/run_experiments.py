"""Train BING and run every experiment reported in the presentation.

Outputs:
  models/bing_*.pkl          trained models
  results/results.json       all numbers (DR / MABO curves, speed, ablations)

Usage:
  python scripts/run_experiments.py                 # synthetic + real images
  python scripts/run_experiments.py --dataset voc --voc-root /path/VOCdevkit/VOC2007
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bing import BING, ProposalEvaluator  # noqa: E402
from bing.datasets import ROOT, get_dataset  # noqa: E402

MAX_N = 5000


# --------------------------------------------------------------- baselines
def random_boxes(img, n=MAX_N, seed=0):
    rng = np.random.default_rng(seed)
    H, W = img.shape[:2]
    x = np.sort(rng.uniform(0, W, (n, 2)), 1)
    y = np.sort(rng.uniform(0, H, (n, 2)), 1)
    return np.stack([x[:, 0], y[:, 0], x[:, 1], y[:, 1]], 1)


def sliding_windows(img, n=MAX_N, seed=0):
    """All windows of the 36 BING sizes on a stride of 1/4 window, random order
    (i.e. what a detector would have to scan without objectness)."""
    H, W = img.shape[:2]
    out = []
    for w in (10, 20, 40, 80, 160, 320):
        for h in (10, 20, 40, 80, 160, 320):
            if w > W or h > H:
                continue
            xs = np.arange(0, W - w + 1, max(w // 4, 1))
            ys = np.arange(0, H - h + 1, max(h // 4, 1))
            X, Y = np.meshgrid(xs, ys)
            X, Y = X.ravel(), Y.ravel()
            out.append(np.stack([X, Y, X + w, Y + h], 1))
    out = np.concatenate(out).astype(float)
    np.random.default_rng(seed).shuffle(out)
    return out[:n]


def selective_search(img, n=MAX_N):
    cv2.setNumThreads(4)
    f = 1.0
    if max(img.shape[:2]) > 500:
        f = 500 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=f, fy=f)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    r = ss.process()[:n].astype(float)
    return np.stack([r[:, 0], r[:, 1], r[:, 0] + r[:, 2], r[:, 1] + r[:, 3]], 1) / f


def interleave(lists, n=MAX_N):
    """BING-Diversified: round-robin merge of ranked lists from several models."""
    out, i = [], 0
    L = max(len(l) for l in lists)
    while len(out) < n and i < L:
        for l in lists:
            if i < len(l):
                out.append(l[i])
        i += 1
    return np.array(out[:n]).reshape(-1, 4)


# --------------------------------------------------------------- evaluation
def evaluate(fn, data, warmup=True):
    ev = ProposalEvaluator(MAX_N)
    if warmup and len(data):
        fn(data[0][0])
    for img, gt in data:
        t = time.perf_counter()
        props = fn(img)
        ev.add(gt, props, time.perf_counter() - t)
    return ev.summary()


def bing_fn(model, **kw):
    return lambda img: model.propose(img, **kw)[0]


def brief(r):
    return (f"DR@100={r['dr_at'][100]:.3f} DR@1000={r['dr_at'][1000]:.3f} "
            f"MABO@1000={r['mabo_at'][1000]:.3f} AUC={r['auc']:.3f} "
            f"{r.get('fps', 0):.1f} fps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "voc"])
    ap.add_argument("--voc-root")
    ap.add_argument("--test-limit", type=int)
    ap.add_argument("--ss-limit", type=int, default=60, help="images for Selective Search")
    ap.add_argument("--skip-ablations", action="store_true")
    a = ap.parse_args()

    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    tag = a.dataset
    train = list(get_dataset(a.dataset, "train", voc_root=a.voc_root))
    test = list(get_dataset(a.dataset, "test", a.test_limit, voc_root=a.voc_root))
    real = list(get_dataset("real", "test"))
    print(f"train {len(train)} imgs / {sum(len(b) for _, b in train)} objs, "
          f"test {len(test)} imgs / {sum(len(b) for _, b in test)} objs, real {len(real)} imgs")

    R = {"dataset": tag, "n_train": len(train), "n_test": len(test), "n_real": len(real)}

    # ------------------------------------------------------------ main model
    t0 = time.time()
    models = {}
    for cs in ("RGB", "HSV", "GRAY"):
        m = BING(colorspace=cs).train(train)
        m.save(os.path.join(ROOT, "models", f"bing_{tag}_{cs.lower()}.pkl"))
        models[cs] = m
    main_m = models["RGB"]
    R["train_seconds"] = time.time() - t0
    R["stage1_train_acc"] = main_m.stage1_acc
    R["w"] = main_m.w.tolist()
    R["calib"] = {f"{w}x{h}": v for (w, h), v in main_m.calib.items()}

    def diversified(img):
        return interleave([models[c].propose(img)[0] for c in ("RGB", "HSV", "GRAY")])

    methods = {
        "BING (binary)": bing_fn(main_m),
        "BING (float w)": None,
        "BING stage I only": bing_fn(main_m, calibrate=False),
        "BING-Diversified": diversified,
        "Random boxes": random_boxes,
        "Sliding windows": sliding_windows,
    }
    for split, data in (("test", test), ("real", real)):
        R[split] = {}
        for name, fn in methods.items():
            if name == "BING (float w)":
                main_m.binary = False
                r = evaluate(bing_fn(main_m), data)
                main_m.binary = True
            else:
                r = evaluate(fn, data)
            R[split][name] = r
            print(f"[{split}] {name:22s} {brief(r)}")
        ss_data = data[:a.ss_limit]
        r = evaluate(selective_search, ss_data, warmup=False)
        r["n_images"] = len(ss_data)
        R[split]["Selective Search (fast)"] = r
        print(f"[{split}] {'Selective Search':22s} {brief(r)} on {len(ss_data)} imgs")
        # BING on the same subset for a like-for-like comparison
        R[split]["BING (binary) @SS subset"] = evaluate(bing_fn(main_m), ss_data)

    # ------------------------------------------------------------ speed
    speed = {}
    sub = test[:100]
    for backend in ("numpy", "numba"):
        for binary in (True, False):
            for interp in ("area", "linear"):
                main_m.backend, main_m.binary, main_m.interp = backend, binary, interp
                fn = bing_fn(main_m)
                fn(sub[0][0])
                t = time.perf_counter()
                for img, _ in sub:
                    fn(img)
                fps = len(sub) / (time.perf_counter() - t)
                key = f"{backend}|{'binary' if binary else 'float'}|{interp}"
                speed[key] = fps
                print(f"[speed] {key:24s} {fps:7.1f} fps")
    main_m.backend, main_m.binary, main_m.interp = "numba", True, "area"
    # per-stage breakdown on one typical image
    img = sub[0][0]
    im = main_m._prep(img)
    H, W = im.shape[:2]
    stages = {"resize": 0.0, "gradient": 0.0, "scoring": 0.0, "nms+topk": 0.0}
    reps = 50
    for _ in range(reps):
        for (w, h) in main_m.calib:
            rw, rh = int(round(8 * W / w)), int(round(8 * H / h))
            if rw < 8 or rh < 8:
                continue
            t = time.perf_counter(); r_ = cv2.resize(im, (rw, rh), interpolation=cv2.INTER_AREA if rw < W else cv2.INTER_LINEAR); stages["resize"] += time.perf_counter() - t
            t = time.perf_counter(); g = main_m.ng_map(r_); stages["gradient"] += time.perf_counter() - t
            t = time.perf_counter(); sm = main_m.score_map(g); stages["scoring"] += time.perf_counter() - t
            t = time.perf_counter(); main_m._nms_topk(sm, main_m.per_size); stages["nms+topk"] += time.perf_counter() - t
    R["speed"] = speed
    R["stage_ms"] = {k: v / reps * 1e3 for k, v in stages.items()}
    print("[stage ms]", R["stage_ms"])

    # ------------------------------------------------------------ ablations
    if not a.skip_ablations:
        abl = {}

        def run(name, m):
            r = evaluate(bing_fn(m), test)
            abl[name] = {k: r[k] for k in ("dr_at", "mabo_at", "auc", "fps", "dr_curve", "curve_n")}
            print(f"[ablation] {name:28s} {brief(r)}")

        for cs in ("RGB", "HSV", "GRAY"):
            run(f"colour={cs}", models[cs])
        run("colour=LAB", BING(colorspace="LAB").train(train, verbose=False))
        run("gradient=Sobel 3x3", BING(kernel="sobel").train(train, verbose=False))
        run("resize=bilinear", BING(interp="linear").train(train, verbose=False))
        for nw in (1, 2, 3, 4):
            m = BING.load(os.path.join(ROOT, "models", f"bing_{tag}_rgb.pkl"), n_basis=nw)
            m._train_stage2(train)
            run(f"Nw={nw}", m)
        for ng in (1, 2, 3, 4, 5, 6):
            m = BING.load(os.path.join(ROOT, "models", f"bing_{tag}_rgb.pkl"), n_bits=ng)
            m._train_stage2(train)
            run(f"Ng={ng}", m)
        for k in (10, 30, 60, 130, 250):
            m = BING.load(os.path.join(ROOT, "models", f"bing_{tag}_rgb.pkl"), per_size=k)
            m._train_stage2(train)
            run(f"per-size={k}", m)
        for n in (10, 50, 150, 300):
            run(f"train imgs={n}", BING().train(train[:n], verbose=False))
        R["ablations"] = abl

    out = os.path.join(ROOT, "results", f"results_{tag}.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
