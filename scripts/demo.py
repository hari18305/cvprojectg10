"""Run BING on any image (or webcam) and visualise the top object proposals.

  python scripts/demo.py --image data/real/coffee.jpg --top 20
  python scripts/demo.py --webcam            # press q to quit; --camera 1 for a second camera
  python scripts/demo.py --list-cameras       # which camera numbers work (e.g. a USB webcam)
  python scripts/demo.py --image photo.jpg --model voc     # model trained on PASCAL VOC 2007

Models: "synthetic" (default) ranks distinct objects highest, so its top 20 boxes make the
clearest picture. "voc" finds more objects within 1000 proposals (96.6% vs 87.9% on our real
photos) but ranks large, near full-image boxes first, as VOC objects are mostly large.
A path to any saved .pkl model also works.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bing import BING  # noqa: E402
from bing.datasets import ROOT  # noqa: E402

MODELS = {
    "synthetic": os.path.join(ROOT, "models", "bing_synthetic_rgb.pkl"),
    "voc": os.path.join(ROOT, "models", "bing_voc_rgb_released_sizes_greedy.pkl"),
}


def draw(img, boxes, scores, top):
    out = img.copy()
    cmap = cv2.applyColorMap(np.linspace(255, 0, top).astype(np.uint8)[:, None], cv2.COLORMAP_JET)[:, 0]
    for i in range(min(top, len(boxes)) - 1, -1, -1):
        x1, y1, x2, y2 = boxes[i].astype(int)
        c = tuple(int(v) for v in cmap[i])
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2 if i < 5 else 1)
        if i < 5:
            cv2.putText(out, f"#{i + 1}", (x1 + 3, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return out


def heat_overlay(img, model):
    heat = model.objectness_heatmap(img)
    hm = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(img, 0.45, hm, 0.55, 0)


def open_camera(index):
    """Open a webcam and check it really delivers frames. On Windows the default
    Media Foundation backend sometimes opens but cannot grab frames, so fall back
    to DirectShow."""
    backends = [cv2.CAP_ANY]
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            for _ in range(10):  # some cameras need a few frames to warm up
                ok, _frame = cap.read()
                if ok:
                    return cap
        cap.release()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--camera", type=int, default=0, help="webcam number (built-in is usually 0, USB 1)")
    ap.add_argument("--list-cameras", action="store_true", help="show which camera numbers deliver frames")
    ap.add_argument("--model", default="synthetic", help="synthetic, voc, or a path to a .pkl model")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "demo"))
    a = ap.parse_args()
    if a.list_cameras:
        found = False
        for i in range(5):
            cap = open_camera(i)
            if cap is not None:
                w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"camera {i}: working ({w}x{h})  ->  python scripts/demo.py --webcam --camera {i}")
                cap.release()
                found = True
        if not found:
            print("No working camera found (checked 0-4).")
        return
    if not a.webcam and not a.image:
        ap.error("give --image PATH or --webcam")
    model_path = MODELS.get(a.model, a.model)
    if not os.path.exists(model_path):
        ap.error(f"model not found: {a.model} (use synthetic, voc, or a .pkl path)")
    model = BING.load(model_path)
    print("model:", os.path.relpath(model_path, ROOT))

    if a.webcam:
        cap = open_camera(a.camera)
        if cap is None:
            sys.exit(f"Could not read frames from camera {a.camera}. Close other apps using it "
                     "(Teams, Zoom, browser), check Windows Settings > Privacy > Camera "
                     "(allow desktop apps), or try --camera 1.")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped sending frames.")
                break
            t = time.perf_counter()
            b, s = model.propose(frame, top=a.top)
            fps = 1 / (time.perf_counter() - t)
            vis = draw(frame, b, s, a.top)
            cv2.putText(vis, f"BING {fps:.0f} fps", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("BING objectness", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
        return

    img = cv2.imread(a.image)
    if img is None:
        ap.error(f"could not read image: {a.image}")
    model.propose(img)  # JIT warm-up
    t = time.perf_counter()
    b, s = model.propose(img)
    dt = time.perf_counter() - t
    print(f"{len(b)} proposals in {dt * 1e3:.1f} ms ({1 / dt:.0f} fps)")
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.image))[0]
    cv2.imwrite(os.path.join(a.out, f"{stem}_top{a.top}.jpg"), draw(img, b, s, a.top))
    cv2.imwrite(os.path.join(a.out, f"{stem}_heat.jpg"), heat_overlay(img, model))
    print("saved to", a.out)


if __name__ == "__main__":
    main()
