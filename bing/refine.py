"""Gradient-guided box refinement: a third stage added to BING (our extension).

BING's boxes sit on a coarse grid (power-of-two sizes, 8-pixel steps in the resized
image), so they rarely fit objects tightly. This stage moves each side of a proposal
to the nearby image line with the strongest edges along that side:

  * left / right sides look for strong horizontal gradients |gx| (vertical edges),
  * top / bottom sides look for strong vertical gradients |gy| (horizontal edges).

Edge strength along any segment is read in O(1) from running sums (integral lines),
so refining 1000 boxes costs a few milliseconds. Ranking and scores are unchanged.
"""
import cv2
import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False


def edge_maps(img):
    """|gx| and |gy| of the image (max over colour channels), float32."""
    f = img.astype(np.float32)
    gx = np.abs(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3))
    if gx.ndim == 3:
        gx, gy = gx.max(2), gy.max(2)
    return gx, gy


def _refine_py(boxes, cx, cy, frac, iters, min_gain):
    H = cx.shape[0] - 1
    W = cy.shape[1] - 1
    out = boxes.copy()
    for i in range(len(out)):
        x1, y1, x2, y2 = out[i]
        for _ in range(iters):
            w, h = x2 - x1, y2 - y1
            if w < 4 or h < 4:
                break
            rx, ry = max(1, int(frac * w)), max(1, int(frac * h))
            ya, yb = int(y1), max(int(y1) + 1, int(y2))
            xa, xb = int(x1), max(int(x1) + 1, int(x2))

            def col(x):  # mean |gx| on column x between rows ya..yb
                return (cx[yb, x] - cx[ya, x]) / (yb - ya)

            def row(y):  # mean |gy| on row y between columns xa..xb
                return (cy[y, xb] - cy[y, xa]) / (xb - xa)

            new = []
            for cur, lo, hi, f in ((int(x1), 0, W - 1, col), (min(int(x2), W - 1), 0, W - 1, col),
                                   (int(y1), 0, H - 1, row), (min(int(y2), H - 1), 0, H - 1, row)):
                r = rx if f is col else ry
                best, bv = cur, f(cur)
                for p in range(max(lo, cur - r), min(hi, cur + r) + 1):
                    v = f(p)
                    if v > bv * (1 + min_gain):
                        best, bv = p, v
                new.append(best)
            nx1, nx2, ny1, ny2 = new
            if nx2 - nx1 < 4 or ny2 - ny1 < 4:
                break
            x1, x2, y1, y2 = float(nx1), float(nx2 + 1), float(ny1), float(ny2 + 1)
        out[i] = (x1, y1, x2, y2)
    return out


if HAVE_NUMBA:
    @njit(cache=True)
    def _best(cur, r, lo, hi, cum, a, b, is_col, min_gain):
        n = max(b - a, 1)
        if is_col:
            bv = (cum[b, cur] - cum[a, cur]) / n
        else:
            bv = (cum[cur, b] - cum[cur, a]) / n
        best = cur
        for p in range(max(lo, cur - r), min(hi, cur + r) + 1):
            if is_col:
                v = (cum[b, p] - cum[a, p]) / n
            else:
                v = (cum[p, b] - cum[p, a]) / n
            if v > bv * (1.0 + min_gain):
                best, bv = p, v
        return best

    @njit(cache=True)
    def _refine_nb(boxes, cx, cy, frac, iters, min_gain):
        H = cx.shape[0] - 1
        W = cy.shape[1] - 1
        out = boxes.copy()
        for i in range(out.shape[0]):
            x1, y1, x2, y2 = out[i, 0], out[i, 1], out[i, 2], out[i, 3]
            for _ in range(iters):
                w, h = x2 - x1, y2 - y1
                if w < 4 or h < 4:
                    break
                rx, ry = max(1, int(frac * w)), max(1, int(frac * h))
                ya, yb = int(y1), max(int(y1) + 1, min(int(y2), H))
                xa, xb = int(x1), max(int(x1) + 1, min(int(x2), W))
                nx1 = _best(int(x1), rx, 0, W - 1, cx, ya, yb, True, min_gain)
                nx2 = _best(min(int(x2), W - 1), rx, 0, W - 1, cx, ya, yb, True, min_gain)
                ny1 = _best(int(y1), ry, 0, H - 1, cy, xa, xb, False, min_gain)
                ny2 = _best(min(int(y2), H - 1), ry, 0, H - 1, cy, xa, xb, False, min_gain)
                if nx2 - nx1 < 4 or ny2 - ny1 < 4:
                    break
                x1, x2, y1, y2 = float(nx1), float(nx2 + 1), float(ny1), float(ny2 + 1)
            out[i, 0], out[i, 1], out[i, 2], out[i, 3] = x1, y1, x2, y2
        return out


def refine_boxes(img, boxes, frac=0.15, iters=2, min_gain=0.0, gx_gy=None):
    """Move each side of each box to the strongest nearby edge line.

    frac: search range per side, as a fraction of the box width / height.
    iters: refinement passes (each pass re-measures edges along the new sides).
    min_gain: a side only moves if the new line's mean edge strength beats the
              current one by this relative margin.
    """
    if len(boxes) == 0:
        return boxes
    gx, gy = gx_gy if gx_gy is not None else edge_maps(img)
    H, W = gx.shape
    cx = np.zeros((H + 1, W), np.float64)  # cx[y, x] = sum of |gx| in column x above row y
    cx[1:] = np.cumsum(gx, axis=0)
    cy = np.zeros((H, W + 1), np.float64)  # cy[y, x] = sum of |gy| in row y left of column x
    cy[:, 1:] = np.cumsum(gy, axis=1)
    b = np.clip(np.asarray(boxes, np.float64), 0, [W, H, W, H])
    if HAVE_NUMBA:
        return _refine_nb(b, cx, cy, float(frac), int(iters), float(min_gain))
    return _refine_py(b, cx, cy, frac, iters, min_gain)
