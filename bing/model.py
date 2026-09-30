"""BING objectness model: two-stage cascaded linear SVM (paper Sec. 3.1-3.2).

Stage I : a single 64-D linear model w scores every 8x8 NG window of the image
          resized to each quantised window size (w_i, h_i).
Stage II: per-size calibration o = v_i * s + t_i so that scores of different
          sizes / aspect ratios become comparable (some sizes, e.g. 10x320,
          rarely contain objects).
"""
import pickle
import time

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from .binary import BinaryScorer
from .fast import HAVE_NUMBA, binary_scores_fast, normed_gradient_fast
from .features import FEAT, box_feature, normed_gradient, resize, to_colorspace
from .metrics import iou_matrix

BASE_SIZES = (10, 20, 40, 80, 160, 320)      # window sizes given in the paper
RELEASED_SIZES = (16, 32, 64, 128, 256, 512)  # sizes used by the authors' released code


class BING:
    def __init__(self, sizes=BASE_SIZES, colorspace="RGB", kernel="simple",
                 binary=True, n_basis=2, n_bits=4, per_size=130, nms_radius=2,
                 C=10.0, interp="area", backend="numba", clip_large=False, seed=0):
        self.sizes = [(w, h) for w in sizes for h in sizes]
        self.colorspace, self.kernel = colorspace, kernel
        self.binary, self.n_basis, self.n_bits = binary, n_basis, n_bits
        self.per_size, self.nms_radius, self.C = per_size, nms_radius, C
        self.interp = interp
        # clip_large: as in the authors' code, keep windows up to 2x the image size and
        # clip them to the image (so near full-image boxes exist); otherwise skip windows
        # more than 1.3x larger than the image.
        self.clip_large = clip_large
        self.backend = backend if HAVE_NUMBA else "numpy"
        self.rng = np.random.default_rng(seed)
        self.w = None
        self.calib = {}  # size -> (v, t)

    # ------------------------------------------------------------------ utils
    def _prep(self, img):
        return to_colorspace(img, self.colorspace)

    def size_index(self, box):
        """Nearest quantised size (in log space) for a box."""
        bw, bh = box[2] - box[0], box[3] - box[1]
        s = np.log2(np.array(BASE_SIZES, float))
        i = int(np.argmin(np.abs(s - np.log2(max(bw, 1)))))
        j = int(np.argmin(np.abs(s - np.log2(max(bh, 1)))))
        return BASE_SIZES[i], BASE_SIZES[j]

    # ------------------------------------------------------------ stage I
    def _stage1_samples(self, dataset, neg_per_img=100):
        X, y = [], []
        for img, boxes in dataset:
            im = self._prep(img)
            H, W = im.shape[:2]
            for b in boxes:
                f = box_feature(im, b, self.kernel, interp=self.interp)
                X.append(f)
                y.append(1)
                X.append(f.reshape(FEAT, FEAT)[:, ::-1].reshape(-1))  # mirror
                y.append(1)
            # random negatives: windows with IoU < 0.5 to every object
            cands = []
            for _ in range(neg_per_img * 3):
                w, h = self.sizes[self.rng.integers(len(self.sizes))]
                if w > W or h > H:
                    continue
                x1 = self.rng.uniform(0, W - w)
                y1 = self.rng.uniform(0, H - h)
                cands.append([x1, y1, x1 + w, y1 + h])
            cands = np.array(cands, float).reshape(-1, 4)
            if len(boxes) and len(cands):
                ok = iou_matrix(np.asarray(boxes, float), cands).max(0) < 0.5
                cands = cands[ok]
            for b in cands[:neg_per_img]:
                X.append(box_feature(im, b, self.kernel, interp=self.interp))
                y.append(0)
        return np.array(X, np.float32), np.array(y)

    def train(self, train_set, verbose=True):
        t0 = time.time()
        X, y = self._stage1_samples(train_set)
        svm = LinearSVC(C=self.C, class_weight="balanced", max_iter=20000,
                        dual=False)
        svm.fit(X / 255.0, y)
        self.w = (svm.coef_[0] / 255.0).astype(np.float32)
        self.stage1_acc = float(svm.score(X / 255.0, y))
        self._build_scorer()
        if verbose:
            print(f"[stage I] {int(y.sum())} pos / {int((1 - y).sum())} neg, "
                  f"train acc {self.stage1_acc:.3f}, {time.time() - t0:.1f}s")
        t0 = time.time()
        self._train_stage2(train_set)
        if verbose:
            print(f"[stage II] calibrated {len(self.calib)} sizes, "
                  f"{time.time() - t0:.1f}s")
        return self

    def _build_scorer(self):
        self.scorer = BinaryScorer(self.w, self.n_basis, self.n_bits) if self.binary else None
        self.kernel_2d = self.w.reshape(FEAT, FEAT)

    # ----------------------------------------------------------- stage II
    def _train_stage2(self, train_set):
        per = {s: ([], []) for s in self.sizes}
        for img, boxes in train_set:
            if not len(boxes):
                continue
            for size, bxs, sc in self._stage1(img):
                lab = iou_matrix(np.asarray(boxes, float), bxs).max(0) >= 0.5
                per[size][0].append(sc)
                per[size][1].append(lab)
        self.calib = {}
        for size, (s, l) in per.items():
            if not s:
                continue
            s, l = np.concatenate(s), np.concatenate(l)
            if l.sum() < 3 or (~l).sum() < 3:
                continue  # size never (or always) holds objects -> discard
            lr = LogisticRegression(C=100.0)
            lr.fit(s[:, None], l)
            self.calib[size] = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))

    # ------------------------------------------------------------ inference
    def _fast(self):
        return self.backend == "numba" and self.kernel == "simple"

    def ng_map(self, im):
        if self._fast():
            return normed_gradient_fast(im)
        return normed_gradient(im, self.kernel)

    def score_map(self, ng):
        if self.binary:
            if self.backend == "numba":
                return binary_scores_fast(ng, self.scorer)
            return self.scorer.score_map(ng)
        return cv2.matchTemplate(ng.astype(np.float32), self.kernel_2d,
                                 cv2.TM_CCORR)

    def _nms_topk(self, sm, k):
        r = self.nms_radius
        dil = cv2.dilate(sm, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
        ys, xs = np.nonzero(sm >= dil)
        sc = sm[ys, xs]
        if len(sc) > k:
            idx = np.argpartition(-sc, k)[:k]
            ys, xs, sc = ys[idx], xs[idx], sc[idx]
        return ys, xs, sc

    def _stage1(self, img, sizes=None):
        """Yield (size, boxes, stage-I scores) for each quantised size."""
        im = self._prep(img)
        H, W = im.shape[:2]
        clip = getattr(self, "clip_large", False)
        for size in (sizes or self.sizes):
            w, h = size
            if clip:
                if w > W * 2 or h > H * 2:
                    continue
                w, h = min(w, W), min(h, H)
            elif w > W * 1.3 or h > H * 1.3:
                continue
            rw, rh = int(round(FEAT * W / w)), int(round(FEAT * H / h))
            if rw < FEAT or rh < FEAT:
                continue
            ng = self.ng_map(resize(im, rw, rh, self.interp))
            sm = self.score_map(ng)
            ys, xs, sc = self._nms_topk(sm, self.per_size)
            sx, sy = W / rw, H / rh
            x1, y1 = xs * sx, ys * sy
            bxs = np.stack([x1, y1, np.minimum(x1 + w, W), np.minimum(y1 + h, H)], 1)
            yield size, bxs, sc  # key = nominal size, so calibration is per quantised size

    def propose(self, img, top=None, calibrate=True, max_side=500):
        """Return (boxes [N,4] as x1,y1,x2,y2, objectness scores) sorted by score.

        Images whose longer side exceeds max_side are first downscaled (PASCAL
        VOC images are ~500 px, matching the 10..320 window sizes) and the
        boxes are mapped back to the original resolution.
        """
        f = 1.0
        if max_side and max(img.shape[:2]) > max_side:
            f = max_side / max(img.shape[:2])
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        sizes = list(self.calib) if calibrate else None
        all_b, all_s = [], []
        for size, bxs, sc in self._stage1(img, sizes):
            if calibrate:
                v, t = self.calib[size]
                sc = v * sc + t
            all_b.append(bxs)
            all_s.append(sc)
        if not all_b:
            return np.zeros((0, 4)), np.zeros(0)
        b, s = np.concatenate(all_b), np.concatenate(all_s)
        o = np.argsort(-s, kind="stable")
        if top:
            o = o[:top]
        return b[o] / f, s[o]

    def objectness_heatmap(self, img, top=2000):
        """Per-pixel sum of objectness of the top proposals (visualisation)."""
        b, s = self.propose(img, top)
        H, W = img.shape[:2]
        b = np.clip(b, 0, [W, H, W, H])
        heat = np.zeros((H, W), np.float32)
        p = 1.0 / (1.0 + np.exp(-s))
        for (x1, y1, x2, y2), v in zip(b.astype(int), p):
            heat[y1:y2, x1:x2] += v
        return heat / max(heat.max(), 1e-6)

    # --------------------------------------------------------------- io
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({k: v for k, v in self.__dict__.items()
                         if k not in ("scorer", "rng")}, f)

    @classmethod
    def load(cls, path, **override):
        m = cls()
        with open(path, "rb") as f:
            m.__dict__.update(pickle.load(f))
        m.__dict__.update(override)
        if not HAVE_NUMBA:
            m.backend = "numpy"
        m._build_scorer()
        return m
