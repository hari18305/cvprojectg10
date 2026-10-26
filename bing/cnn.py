"""CNN features for proposal windows (Fast R-CNN style, frozen ImageNet backbone).

The handcrafted detector (bing/detector.py) is limited by HOG. Here each image goes once
through a ResNet-50 pre-trained on ImageNet (no fine-tuning), and every proposal window
is described by the mean conv4 and conv5 activations inside it (read from integral images
of the feature maps, so pooling 1000 boxes costs a few milliseconds). A linear classifier per
VOC class and a class-specific box regressor (as in R-CNN) are trained on top.

Only the detector uses a deep network; the proposals still come from BING.
"""
import cv2
import numpy as np

import torch
import torchvision

from .detector import VOC_CLASSES, nms

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class CNNFeatures:
    """Pooled conv4 + conv5 features of boxes, from one backbone pass per image."""

    def __init__(self, max_side=600):
        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        net = torchvision.models.resnet50(weights=w).eval()
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2)
        self.c4, self.c5 = net.layer3, net.layer4
        self.max_side = max_side
        self.dim = 1024 + 2048

    @torch.no_grad()
    def maps(self, img):
        """Integral images of the conv4 and conv5 maps of a BGR image, plus the scale
        applied to box coordinates."""
        s = self.max_side / max(img.shape[:2])
        im = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA)
        x = (im[..., ::-1].astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]
        f4 = self.c4(self.stem(x))
        f5 = self.c5(f4)
        return _integral(f4[0].numpy()), _integral(f5[0].numpy()), s

    def pool(self, maps, boxes):
        """[N, 3072] features: the mean of the conv4 (stride 16) and conv5 (stride 32)
        activations inside each box, read from integral images in O(1) per box,
        then square-rooted and L2-normalised."""
        I4, I5, s = maps
        if len(boxes) == 0:
            return np.zeros((0, self.dim), np.float32)
        b = np.asarray(boxes, np.float32) * s
        f = np.concatenate([_box_mean(I4, b, 16), _box_mean(I5, b, 32)], 1)
        f = np.sqrt(np.maximum(f, 0))
        return (f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-6)).astype(np.float32)


def _integral(fm):
    """[C, H, W] -> [H+1, W+1, C] running sums (float64)."""
    C, H, W = fm.shape
    I = np.zeros((H + 1, W + 1, C), np.float64)
    I[1:, 1:] = fm.transpose(1, 2, 0).cumsum(0).cumsum(1)
    return I


def _box_mean(I, b, stride):
    H, W = I.shape[0] - 1, I.shape[1] - 1
    x1 = np.clip(np.floor(b[:, 0] / stride), 0, W - 1).astype(int)
    y1 = np.clip(np.floor(b[:, 1] / stride), 0, H - 1).astype(int)
    x2 = np.clip(np.ceil(b[:, 2] / stride), x1 + 1, W).astype(int)
    y2 = np.clip(np.ceil(b[:, 3] / stride), y1 + 1, H).astype(int)
    s = I[y2, x2] - I[y1, x2] - I[y2, x1] + I[y1, x1]
    return (s / ((x2 - x1) * (y2 - y1))[:, None]).astype(np.float32)


def train_linear(X, Y, M, lam=1e-4, iters=300, pos_weight=3.0):
    """One-vs-rest linear SVMs (squared hinge, L2) for all classes at once.
    X [N, D] features, Y [N, K] in {-1, +1}, M [N, K] sample weights (0 = not used for that
    class). Returns W [K, D], b [K]."""
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y.astype(np.float32))
    M = torch.from_numpy(M.astype(np.float32)) * torch.where(Y > 0, pos_weight, 1.0)
    K, D = Y.shape[1], X.shape[1]
    W = torch.zeros(D, K, requires_grad=True)
    b = torch.zeros(K, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")
    norm = M.sum(0).clamp_min(1)

    def closure():
        opt.zero_grad()
        m = torch.clamp(1 - Y * (X @ W + b), min=0) ** 2
        loss = ((m * M).sum(0) / norm).sum() + lam * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach().numpy().T.copy(), b.detach().numpy().copy()


def box_targets(props, gts):
    """R-CNN regression targets (dx, dy, dw, dh) from proposals to their ground-truth boxes."""
    pw, ph = props[:, 2] - props[:, 0], props[:, 3] - props[:, 1]
    px, py = props[:, 0] + pw / 2, props[:, 1] + ph / 2
    gw, gh = gts[:, 2] - gts[:, 0], gts[:, 3] - gts[:, 1]
    gx, gy = gts[:, 0] + gw / 2, gts[:, 1] + gh / 2
    return np.stack([(gx - px) / pw, (gy - py) / ph, np.log(gw / pw), np.log(gh / ph)], 1)


def apply_deltas(boxes, d):
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    cx, cy = boxes[:, 0] + w / 2 + d[:, 0] * w, boxes[:, 1] + h / 2 + d[:, 1] * h
    w, h = w * np.exp(np.clip(d[:, 2], -2, 2)), h * np.exp(np.clip(d[:, 3], -2, 2))
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def fit_ridge(X, T, lam=1.0):
    """Ridge regression T ≈ X @ A + c, solved in closed form."""
    mu = X.mean(0)
    Xc = X - mu
    A = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1], dtype=np.float64), Xc.T @ (T - T.mean(0)))
    return A.astype(np.float32), (T.mean(0) - mu @ A).astype(np.float32)


class CNNDetector:
    """Linear classifiers + box regressors on CNN window features."""

    def __init__(self, W, b, R=None, r0=None):
        self.classes = list(VOC_CLASSES)
        self.W, self.b, self.R, self.r0 = W, b, R, r0  # R [K, D, 4], r0 [K, 4]

    def scores(self, f):
        return f @ self.W.T + self.b

    def detect(self, boxes, feats, regress=True, nms_iou=0.3, min_score=-1.0, per_class=50):
        """{class index: [K, 5] (score, x1, y1, x2, y2)} after per-class NMS."""
        sc = self.scores(feats)
        out = {}
        for j in range(len(self.classes)):
            o = np.nonzero(sc[:, j] > min_score)[0]
            if not len(o):
                continue
            o = o[np.argsort(-sc[o, j])][: per_class * 4]
            bx = boxes[o]
            if regress and self.R is not None:
                bx = apply_deltas(bx, feats[o] @ self.R[j] + self.r0[j])
            keep = nms(bx, sc[o, j], nms_iou)[:per_class]
            out[j] = np.c_[sc[o, j][keep], bx[keep]].astype(np.float32)
        return out
