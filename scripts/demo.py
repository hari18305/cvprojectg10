"""Run BING on any image (or webcam) and visualise the top object proposals.

  python scripts/demo.py --image data/real/coffee.jpg --top 20
  python scripts/demo.py --webcam            # press q to quit; --camera 1 for a second camera
  python scripts/demo.py --list-cameras       # which camera numbers work (e.g. a USB webcam)
  python scripts/demo.py --image photo.jpg --model voc     # model trained on PASCAL VOC 2007
  python scripts/demo.py --image photo.jpg --rerank        # + our multi-cue re-ranking (stage III)
  python scripts/demo.py --image photo.jpg --compare       # BING vs re-ranked, side by side
  python scripts/demo.py --image photo.jpg --detect        # + our detector (20 VOC classes; CNN features
                                                           #   if torch is installed, else HOG)

Models: "synthetic" (default) ranks distinct objects highest, so its top 20 boxes make the
clearest picture. "voc" finds more objects within 1000 proposals (96.6% vs 87.9% on our real
photos) but ranks large, near full-image boxes first, as VOC objects are mostly large.
A path to any saved .pkl model also works.
"""
import argparse
import os
import pickle
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


def tidy(dets, max_n=5, margin=0.12):
    """Keep the best few detections for display: within `margin` of the best score of
    their class, and not overlapping a kept box of the same class (IoU > 0.2) or of any
    class (IoU > 0.6)."""
    from bing.metrics import iou_matrix
    best = {}
    for name, sc, _ in dets:
        best[name] = max(best.get(name, -1e9), sc)
    kept = []
    for name, sc, box in dets:
        if sc < best[name] - margin:
            continue
        if all(iou_matrix([box], [k[2]])[0, 0] <= (0.2 if name == k[0] else 0.6) for k in kept):
            kept.append((name, sc, box))
        if len(kept) == max_n:
            break
    return kept


def draw_detections(img, dets):
    """dets: [(class name, score, box)] sorted by score."""
    out = img.copy()
    H, W = img.shape[:2]
    fs = max(0.5, min(W, H) / 600)  # label size grows with the image
    for name, sc, (x1, y1, x2, y2) in tidy(dets):
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(W - 1, x2)), int(min(H - 1, y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), max(2, int(2 * fs)))
        label = f"{name} {sc:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        lx = min(x1, W - tw - 6)
        cv2.rectangle(out, (lx, y1), (lx + tw + 6, y1 + th + 8), (0, 220, 0), -1)
        cv2.putText(out, label, (lx + 3, y1 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)
    return out


class Pipeline:
    """BING, optionally followed by our re-ranking (stage III) and our detector."""

    def __init__(self, model, rerank=False, detect=None, windows=100, min_score=0.0):
        self.model, self.reranker, self.detector, self.cnn = model, None, None, None
        self.windows, self.min_score = windows, min_score
        if rerank or detect:
            with open(os.path.join(ROOT, "models", "rerank_voc.pkl"), "rb") as f:
                self.reranker = pickle.load(f)
        cnn_path = os.path.join(ROOT, "models", "cnn_detector_voc.pkl")
        if detect == "auto":
            detect = "cnn" if os.path.exists(cnn_path) and _have_torch() else "hog"
        self.kind = detect
        if self.min_score is None:
            self.min_score = -0.4 if detect == "cnn" else 0.0
        if detect == "cnn":
            from bing.cnn import CNNFeatures  # downloads the ResNet-50 weights on first use
            self.cnn = CNNFeatures()
            with open(cnn_path, "rb") as f:
                self.detector = pickle.load(f)
        elif detect == "hog":
            with open(os.path.join(ROOT, "models", "detector_voc.pkl"), "rb") as f:
                self.detector = pickle.load(f)

    def __call__(self, img, top=None):
        b, s = self.model.propose(img)
        if self.reranker is not None:
            b, s = self.reranker.rerank(img, b, s)
        dets = None
        if self.detector is not None:
            w = b[: self.windows]
            if self.cnn is not None:
                per = self.detector.detect(w, self.cnn.pool(self.cnn.maps(img), w), min_score=self.min_score)
            else:
                from bing.detector import box_features
                per = self.detector.detect(w, self.detector.scores(box_features(img, w)), min_score=self.min_score)
            dets = sorted(((self.detector.classes[j], float(d[0]), d[1:]) for j, a in per.items() for d in a),
                          key=lambda t: -t[1])
        return (b[:top] if top else b), s, dets


def _have_torch():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return True
    except ImportError:
        return False


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
    ap.add_argument("--rerank", action="store_true", help="add our multi-cue re-ranking (uses the VOC model)")
    ap.add_argument("--compare", action="store_true", help="image only: BING vs re-ranked top boxes side by side")
    ap.add_argument("--detect", nargs="?", const="auto", choices=["auto", "cnn", "hog"],
                    help="add our detector for the 20 VOC classes on the top --windows boxes: cnn (ResNet-50 "
                         "features, needs torch), hog, or auto (cnn when available)")
    ap.add_argument("--windows", type=int, default=100, help="boxes the detector looks at (default 100)")
    ap.add_argument("--min-score", type=float, default=None,
                    help="detector score threshold (default: -0.4 for cnn, 0 for hog)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "demo"))
    ap.add_argument("--show", action="store_true", help="image mode: also open the results in a window")
    a = ap.parse_args()
    if a.list_cameras:
        try:  # probing missing camera numbers makes OpenCV print warnings; hide them
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except AttributeError:
            pass
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
    if (a.rerank or a.detect or a.compare) and a.model == "synthetic":
        a.model = "voc"  # the re-ranker and the detector were trained on BING's VOC proposals
    model_path = MODELS.get(a.model, a.model)
    if not os.path.exists(model_path):
        ap.error(f"model not found: {a.model} (use synthetic, voc, or a .pkl path)")
    model = BING.load(model_path, n_jobs=min(4, os.cpu_count() or 1))  # window sizes in parallel threads
    print("model:", os.path.relpath(model_path, ROOT))
    pipe = Pipeline(model, a.rerank or a.compare, a.detect, a.windows, a.min_score)
    tag = "BING" + (" + re-ranking" if a.rerank else "") + (f" + {pipe.kind.upper()} detector" if a.detect else "")

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
            b, s, dets = pipe(frame, top=a.top)
            fps = 1 / (time.perf_counter() - t)
            vis = draw_detections(frame, dets) if dets is not None else draw(frame, b, s, a.top)
            cv2.putText(vis, f"{tag}: {fps:.0f} fps", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("BING objectness", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
        return

    img = cv2.imread(a.image)
    if img is None:
        ap.error(f"could not read image: {a.image}")
    pipe(img)  # JIT warm-up
    t = time.perf_counter()
    b, s, dets = pipe(img)
    dt = time.perf_counter() - t
    print(f"{tag}: {len(b)} proposals in {dt * 1e3:.1f} ms ({1 / dt:.0f} fps)")
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.image))[0]
    saved = []
    if a.compare:
        b0, s0 = model.propose(img)
        left, right = draw(img, b0, s0, a.top), draw(img, b, s, a.top)
        for v, t_ in ((left, f"BING: top {a.top}"), (right, f"+ re-ranking: top {a.top}")):
            cv2.putText(v, t_, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(v, t_, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        saved.append((f"{stem}_compare.jpg", np.hstack([left, right])))
    elif dets is not None:
        for name, sc, _ in tidy(dets):
            print(f"  {name:12s} {sc:.2f}")
        saved.append((f"{stem}_detections.jpg", draw_detections(img, dets)))
    else:
        suffix = "_reranked" if a.rerank else ""
        saved.append((f"{stem}_top{a.top}{suffix}.jpg", draw(img, b, s, a.top)))
        saved.append((f"{stem}_heat.jpg", heat_overlay(img, model)))
    for name, v in saved:
        cv2.imwrite(os.path.join(a.out, name), v)
        print("saved", os.path.join(a.out, name))
    if a.show:
        for name, v in saved:
            cv2.imshow(name, v)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
