"""Controlled speed benchmark (run on an otherwise idle machine).

Updates results/results_synthetic.json (speed + per-stage time, 4 threads and 1 thread)
and, if --voc-root is given, results/voc.json (our final VOC model, 1 and 4 threads).
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bing import BING  # noqa: E402
from bing.datasets import ROOT, get_dataset  # noqa: E402

try:
    import numba
except ImportError:  # pragma: no cover
    numba = None


def set_threads(n):
    cv2.setNumThreads(n)
    if numba is not None:
        numba.set_num_threads(n)


def fps(model, images, reps=1):
    model.propose(images[0])  # warm-up / JIT
    t = time.perf_counter()
    for _ in range(reps):
        for img in images:
            model.propose(img)
    return reps * len(images) / (time.perf_counter() - t)


def stage_breakdown(model, img, reps=50):
    im = model._prep(img)
    H, W = im.shape[:2]
    st = {"resize": 0.0, "gradient": 0.0, "scoring": 0.0, "nms+topk": 0.0}
    for _ in range(reps):
        for (w, h) in model.calib:
            rw, rh = int(round(8 * W / w)), int(round(8 * H / h))
            if w > W * 1.3 or h > H * 1.3 or rw < 8 or rh < 8:
                continue
            t = time.perf_counter(); r = cv2.resize(im, (rw, rh), interpolation=cv2.INTER_AREA if rw < W else cv2.INTER_LINEAR); st["resize"] += time.perf_counter() - t
            t = time.perf_counter(); g = model.ng_map(r); st["gradient"] += time.perf_counter() - t
            t = time.perf_counter(); sm = model.score_map(g); st["scoring"] += time.perf_counter() - t
            t = time.perf_counter(); model._nms_topk(sm, model.per_size); st["nms+topk"] += time.perf_counter() - t
    return {k: v / reps * 1e3 for k, v in st.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc-root")
    a = ap.parse_args()

    path = os.path.join(ROOT, "results", "results_synthetic.json")
    R = json.load(open(path))
    model = BING.load(os.path.join(ROOT, "models", "bing_synthetic_rgb.pkl"))
    imgs = [img for img, _ in get_dataset("synthetic", "test", 100)]

    set_threads(4)
    speed = {}
    for backend in ("numpy", "numba"):
        for binary in (True, False):
            for interp in ("area", "linear"):
                model.backend, model.binary, model.interp = backend, binary, interp
                key = f"{backend}|{'binary' if binary else 'float'}|{interp}"
                speed[key] = fps(model, imgs)
                print(f"[speed, 4 threads] {key:24s} {speed[key]:7.1f} img/s", flush=True)
    model.backend, model.binary, model.interp = "numba", True, "area"
    R["stage_ms"] = stage_breakdown(model, imgs[0])
    print("[stage ms]", R["stage_ms"])
    set_threads(1)
    model.interp = "linear"
    speed["numba|binary|linear|1thread"] = fps(model, imgs)
    print(f"[speed, 1 thread] numba|binary|linear {speed['numba|binary|linear|1thread']:.1f} img/s")
    set_threads(4)
    R["speed"] = speed
    json.dump(R, open(path, "w"), indent=1)

    if a.voc_root:
        vpath = os.path.join(ROOT, "results", "voc.json")
        V = json.load(open(vpath))
        vm = BING.load(os.path.join(ROOT, "models", "bing_voc_rgb_released_sizes_greedy.pkl"))
        vimgs = [cv2.imread(os.path.join(a.voc_root, "JPEGImages", i + ".jpg")) for i in V["ss_subset_ids"]]
        V["speed_ours"] = {}
        for n in (4, 1):
            set_threads(n)
            V["speed_ours"][f"{n} thread{'s' if n > 1 else ''}"] = f_ = fps(vm, vimgs)
            print(f"[VOC speed, {n} thread(s)] {f_:.1f} img/s")
        json.dump(V, open(vpath, "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    main()
