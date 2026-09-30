"""Object-proposal evaluation metrics: DR-#WIN curves, MABO, AUC."""
import numpy as np


def iou_matrix(a, b):
    """IoU between boxes a [N,4] and b [M,4] (x1,y1,x2,y2) -> [N,M]."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None] - inter, 1e-9)


def best_overlap_curve(gt, props, max_n):
    """best[g, n-1] = best IoU of GT g among the first n proposals."""
    props = props[:max_n]
    out = np.zeros((len(gt), max_n))
    if len(props) == 0 or len(gt) == 0:
        return out
    iou = iou_matrix(gt, props)
    cum = np.maximum.accumulate(iou, axis=1)
    out[:, :cum.shape[1]] = cum
    if cum.shape[1] < max_n:  # fewer proposals than max_n: curve saturates
        out[:, cum.shape[1]:] = cum[:, -1:]
    return out


class ProposalEvaluator:
    """Accumulates per-object best overlaps over a dataset."""

    def __init__(self, max_n=5000):
        self.max_n = max_n
        self.curves = []
        self.times = []
        self.counts = []

    def add(self, gt, props, seconds=None):
        if len(gt):
            self.curves.append(best_overlap_curve(np.asarray(gt, float), props, self.max_n))
        self.counts.append(len(props))
        if seconds is not None:
            self.times.append(seconds)

    def summary(self, thr=0.5, ns=(1, 10, 100, 500, 1000, 2000, 5000)):
        B = np.concatenate(self.curves, 0)  # [num_objects, max_n]
        dr = (B >= thr).mean(0)
        mabo = B.mean(0)
        xs = np.unique(np.round(np.logspace(0, np.log10(self.max_n), 200)).astype(int)) - 1
        res = {
            "num_objects": int(B.shape[0]),
            "avg_proposals": float(np.mean(self.counts)),
            "dr_curve": dr[xs].tolist(),
            "mabo_curve": mabo[xs].tolist(),
            "curve_n": (xs + 1).tolist(),
            "dr_at": {int(n): float(dr[min(n, self.max_n) - 1]) for n in ns},
            "mabo_at": {int(n): float(mabo[min(n, self.max_n) - 1]) for n in ns},
            # area under DR-#WIN curve on log axis, normalised to [0,1]
            "auc": float(np.mean(dr[xs])),
            # DR vs IoU threshold at 1000 proposals (recall-IoU curve)
            "recall_iou": {f"{t:.2f}": float((B[:, min(1000, self.max_n) - 1] >= t).mean())
                           for t in np.arange(0.5, 1.0, 0.05)},
        }
        if self.times:
            t = float(np.mean(self.times))
            res["sec_per_img"] = t
            res["fps"] = 1.0 / t if t > 0 else float("inf")
        return res
