"""A complete object detector built on BING proposals (R-CNN-style, handcrafted features).

BING only says *where* objects might be. To show what proposals are for, each proposal
is warped to 64x64 and described with handcrafted features (HOG + a hue/saturation
histogram); one linear SVM per PASCAL VOC class scores it, and per-class non-maximum
suppression gives the final detections. Accuracy is measured with the VOC 2007
average precision (11-point AP, 'difficult' objects ignored).
"""
import cv2
import numpy as np
from sklearn.svm import LinearSVC

from .metrics import iou_matrix

VOC_CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor")
PATCH = 64
_HOG = cv2.HOGDescriptor((PATCH, PATCH), (16, 16), (8, 8), (8, 8), 9)
N_HOG = _HOG.getDescriptorSize()  # 1764
N_COL = 8 * 4


def _describe(patch):
    hog = _HOG.compute(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)).ravel()
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    col = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256]).ravel()
    col = np.sqrt(col / max(col.sum(), 1.0))
    return np.concatenate([hog, col])


def box_features(img, boxes, flip=False):
    """Features [N, 1796] of boxes (x1, y1, x2, y2) warped to 64x64 (with 16 px of context
    at that scale, as in R-CNN). flip=True also returns the mirrored patches' features."""
    H, W = img.shape[:2]
    out = np.empty((len(boxes) * (2 if flip else 1), N_HOG + N_COL), np.float32)
    k = 0
    for x1, y1, x2, y2 in np.asarray(boxes, float):
        # pad by 1/8 of the box on each side, so the object fills the central 48x48
        pw, ph = (x2 - x1) / 8, (y2 - y1) / 8
        a, b = int(max(0, x1 - pw)), int(max(0, y1 - ph))
        c, d = int(min(W, np.ceil(x2 + pw))), int(min(H, np.ceil(y2 + ph)))
        crop = img[b:max(d, b + 1), a:max(c, a + 1)]
        patch = cv2.resize(crop, (PATCH, PATCH), interpolation=cv2.INTER_LINEAR)
        out[k] = _describe(patch)
        k += 1
        if flip:
            out[k] = _describe(np.ascontiguousarray(patch[:, ::-1]))
            k += 1
    return out


class Detector:
    """One linear SVM per class on handcrafted features of proposal windows."""

    def __init__(self, classes=VOC_CLASSES, C=0.01):
        self.classes, self.C = list(classes), C
        self.W = None  # [n_classes, D]
        self.b = None

    def fit_class(self, X_pos, X_neg):
        X = np.concatenate([X_pos, X_neg])
        y = np.r_[np.ones(len(X_pos)), np.zeros(len(X_neg))]
        svm = LinearSVC(C=self.C, class_weight={1: 2.0, 0: 1.0}, dual=False, max_iter=5000)
        svm.fit(X, y)
        return svm.coef_[0].astype(np.float32), float(svm.intercept_[0])

    def scores(self, feats):
        return feats @ self.W.T + self.b  # [N, n_classes]

    def detect(self, boxes, scores, nms_iou=0.3, min_score=-1.0, per_class=50):
        """Per-class NMS over windows scoring above min_score.
        Returns {class index: array [K, 5] of (score, x1, y1, x2, y2)}."""
        out = {}
        for j in range(len(self.classes)):
            s = scores[:, j]
            o = np.nonzero(s > min_score)[0]
            if not len(o):
                continue
            o = o[np.argsort(-s[o])][: per_class * 4]
            keep = nms(boxes[o], s[o], nms_iou)[:per_class]
            out[j] = np.c_[s[o][keep], boxes[o][keep]].astype(np.float32)
        return out


def nms(boxes, scores, thr):
    """Greedy non-maximum suppression; boxes must already be sorted by score."""
    if len(boxes) == 0:
        return np.zeros(0, int)
    iou = iou_matrix(boxes, boxes)
    alive = np.ones(len(boxes), bool)
    keep = []
    for i in range(len(boxes)):
        if not alive[i]:
            continue
        keep.append(i)
        alive &= iou[i] <= thr
    return np.array(keep, int)


def voc_ap(rec, prec):
    """PASCAL VOC 2007 11-point interpolated average precision."""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p = prec[rec >= t].max() if np.any(rec >= t) else 0.0
        ap += p / 11.0
    return ap


def average_precision(dets, gts):
    """dets: array [K, 6] of (image index, score, x1, y1, x2, y2);
    gts: {image index: (boxes [M,4], difficult [M] bool)}.
    A detection is correct if IoU >= 0.5 with an unmatched ground-truth box; matches to
    'difficult' boxes are ignored, as in the VOC devkit."""
    n_pos = sum(int((~d).sum()) for _, d in gts.values())
    if n_pos == 0:
        return float("nan")
    dets = dets[np.argsort(-dets[:, 1], kind="stable")]
    used = {k: np.zeros(len(v[0]), bool) for k, v in gts.items()}
    tp, fp = np.zeros(len(dets)), np.zeros(len(dets))
    for i, row in enumerate(dets):
        img = int(row[0])
        g, diff = gts.get(img, (np.zeros((0, 4)), np.zeros(0, bool)))
        if len(g) == 0:
            fp[i] = 1
            continue
        ov = voc_iou(row[2:], g)
        j = int(np.argmax(ov))
        if ov[j] >= 0.5:
            if diff[j]:
                continue
            if not used[img][j]:
                tp[i], used[img][j] = 1, True
            else:
                fp[i] = 1
        else:
            fp[i] = 1
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / n_pos
    prec = tp / np.maximum(tp + fp, np.finfo(float).eps)
    return voc_ap(rec, prec)


def voc_iou(box, gts):
    """IoU of one box against many. Boxes are 0-based with exclusive right/bottom edges,
    which gives the same areas as the devkit's +1 rule on its 1-based inclusive boxes."""
    ix1 = np.maximum(gts[:, 0], box[0])
    iy1 = np.maximum(gts[:, 1], box[1])
    ix2 = np.minimum(gts[:, 2], box[2])
    iy2 = np.minimum(gts[:, 3], box[3])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
    return inter / (a + b - inter)
