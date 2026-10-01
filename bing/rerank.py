"""Multi-cue re-ranking of BING proposals: a third stage added to BING (our extension).

BING ranks windows with one cue, the 8x8 edge template. This stage adds cheap
handcrafted cues, all read from integral images in O(1) per window:

  * colour contrast: Lab colour inside the window vs. a surrounding ring,
  * edge density: gradient energy inside the window vs. the ring,
  * geometry: window centre, size and aspect ratio relative to the image,
  * BING's own calibrated score and rank.

A small gradient-boosted tree model, trained on proposals from training images
(label = IoU >= 0.5 with a ground-truth object), combines them into a new score,
and the proposals are re-sorted. No deep network is involved.
"""
import cv2
import numpy as np


def _integral(ch):
    return cv2.integral(ch.astype(np.float64))  # (H+1, W+1)


def _box_sum(I, x1, y1, x2, y2):
    return I[y2, x2] - I[y1, x2] - I[y2, x1] + I[y1, x1]


def features(img, boxes, scores):
    """Feature matrix [N, F] for proposals (boxes in img coordinates)."""
    H, W = img.shape[:2]
    n = len(boxes)
    if n == 0:
        return np.zeros((0, 16), np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mag = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    Is = [_integral(lab[..., c]) for c in range(3)] + [_integral(mag)]
    b = np.clip(np.round(np.asarray(boxes, float)).astype(int), 0, [W, H, W, H])
    x1, y1, x2, y2 = b.T
    x2 = np.minimum(np.maximum(x2, x1 + 1), W)
    y2 = np.minimum(np.maximum(y2, y1 + 1), H)
    x1 = np.minimum(x1, x2 - 1)
    y1 = np.minimum(y1, y2 - 1)
    bw, bh = x2 - x1, y2 - y1
    # surrounding ring: the box grown by half its size on each axis, clipped to the image
    ox1 = np.clip(x1 - bw // 4, 0, W); oy1 = np.clip(y1 - bh // 4, 0, H)
    ox2 = np.clip(x2 + bw // 4, 0, W); oy2 = np.clip(y2 + bh // 4, 0, H)
    a_in = (bw * bh).astype(float)
    a_out = ((ox2 - ox1) * (oy2 - oy1)).astype(float)
    a_ring = np.maximum(a_out - a_in, 1.0)
    means_in, means_ring = [], []
    for I in Is:
        s_in = _box_sum(I, x1, y1, x2, y2)
        s_out = _box_sum(I, ox1, oy1, ox2, oy2)
        means_in.append(s_in / a_in)
        means_ring.append((s_out - s_in) / a_ring)
    m_in, m_ring = np.stack(means_in, 1), np.stack(means_ring, 1)
    colour_contrast = np.linalg.norm(m_in[:, :3] - m_ring[:, :3], axis=1)
    edge_in, edge_ring = m_in[:, 3], m_ring[:, 3]
    ring_share = (a_out - a_in) / a_in  # 0 when the box already fills the image
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    rank = np.arange(n)
    touches = ((x1 == 0) | (y1 == 0) | (x2 == W) | (y2 == H)).astype(float)
    return np.stack([
        scores, np.log1p(rank),
        colour_contrast, m_in[:, 0] - m_ring[:, 0],
        edge_in, edge_ring, np.log((edge_in + 1) / (edge_ring + 1)),
        cx, cy, (cx - .5) ** 2 + (cy - .5) ** 2,
        np.log(a_in / (W * H)), np.log(bw / bh), bw / W, bh / H,
        ring_share, touches,
    ], 1).astype(np.float32)


class Reranker:
    def __init__(self, max_iter=200, seed=0):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.1,
                                                  max_leaf_nodes=31, random_state=seed)

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def rerank(self, img, boxes, scores):
        """Return (boxes, new scores) sorted by the re-ranker's probability."""
        if len(boxes) == 0:
            return boxes, scores
        p = self.clf.predict_proba(features(img, boxes, scores))[:, 1]
        o = np.argsort(-p, kind="stable")
        return boxes[o], p[o]
