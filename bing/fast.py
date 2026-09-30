"""JIT-compiled (Numba) fast path mirroring the paper's Algorithm 2.

For every pixel the 8-bit normed gradient is computed, its top Ng bits are
pushed into per-row bytes with a single shift/OR, row bytes are stacked into a
64-bit word with another shift/OR, and window scores are obtained with
BITWISE AND + POPCOUNT only.  Falls back to the numpy implementation when
Numba is not installed.
"""
import numpy as np

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

if HAVE_NUMBA:
    @njit(cache=True, inline="always")
    def _popcount(x):
        x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
        x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
        x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)

    @njit(cache=True)
    def _ng(img):
        """Normed gradient ([-1,0,1] mask, max over channels) of H x W x C uint8."""
        H, W, C = img.shape
        g = np.empty((H, W), np.uint8)
        for y in range(H):
            y0 = y - 1 if y > 0 else y
            y1 = y + 1 if y < H - 1 else y
            fy = 2 if (y == 0 or y == H - 1) else 1
            for x in range(W):
                x0 = x - 1 if x > 0 else x
                x1 = x + 1 if x < W - 1 else x
                fx = 2 if (x == 0 or x == W - 1) else 1
                best_x = 0
                best_y = 0
                for c in range(C):
                    dx = abs(np.int32(img[y, x1, c]) - np.int32(img[y, x0, c])) * fx
                    dy = abs(np.int32(img[y1, x, c]) - np.int32(img[y0, x, c])) * fy
                    if dx > best_x:
                        best_x = dx
                    if dy > best_y:
                        best_y = dy
                v = best_x + best_y
                g[y, x] = 255 if v > 255 else v
        return g

    @njit(cache=True, parallel=True)
    def _binary_scores(g, a_plus, betas, n_bits):
        H, W = g.shape
        oh, ow = H - 7, W - 7
        nw = a_plus.shape[0]
        # pass 1: row bytes r[k, y, x] = 8 horizontal bits of bit-plane k (shift/OR)
        r = np.zeros((n_bits, H, ow), np.uint64)
        for y in prange(H):
            for k in range(n_bits):
                shift = 7 - k
                row = np.uint64(0)
                for x in range(W):
                    bit = np.uint64((g[y, x] >> shift) & 1)
                    row = ((row << np.uint64(1)) | bit) & np.uint64(0xFF)
                    if x >= 7:
                        r[k, y, x - 7] = row
        # pass 2+3: stack 8 row bytes into a 64-bit word, score with AND+POPCOUNT
        out = np.empty((oh, ow), np.float32)
        for y in prange(oh):
            for x in range(ow):
                s = 0.0
                for k in range(n_bits):
                    b = np.uint64(0)
                    for j in range(8):
                        b = (b << np.uint64(8)) | r[k, y + j, x]
                    nb = np.int64(_popcount(b))
                    acc = 0.0
                    for j in range(nw):
                        acc += betas[j] * (2 * np.int64(_popcount(b & a_plus[j])) - nb)
                    s += acc * (1 << (7 - k))
                out[y, x] = s
        return out


if HAVE_NUMBA:
    @njit(cache=True)
    def _greedy_nms(sm, order, radius, k):
        """The authors' non-maximum suppression: visit windows from the highest score
        down, accept a window unless an accepted one lies within `radius` map cells,
        stop after k windows."""
        H, W = sm.shape
        blocked = np.zeros((H, W), np.bool_)
        ys = np.empty(k, np.int64)
        xs = np.empty(k, np.int64)
        n = 0
        for idx in order:
            y, x = idx // W, idx % W
            if blocked[y, x]:
                continue
            ys[n], xs[n] = y, x
            n += 1
            if n >= k:
                break
            for yy in range(max(0, y - radius), min(H, y + radius + 1)):
                for xx in range(max(0, x - radius), min(W, x + radius + 1)):
                    blocked[yy, xx] = True
        return ys[:n], xs[:n]


def greedy_nms(sm, radius, k):
    flat = sm.ravel()
    # Greedy suppression only ever needs the best few thousand windows: sort those first
    # and fall back to a full sort only if they do not yield k accepted windows.
    m = k * (2 * radius + 1) ** 2
    if HAVE_NUMBA and flat.size > 2 * m:
        top = np.argpartition(-flat, m)[:m]
        order = top[np.argsort(-flat[top], kind="stable")]
        ys, xs = _greedy_nms(sm, order, radius, k)
        if len(ys) >= k or len(ys) == 0:
            return ys, xs, sm[ys, xs]
    order = np.argsort(-flat, kind="stable")
    if HAVE_NUMBA:
        ys, xs = _greedy_nms(sm, order, radius, k)
    else:  # plain Python fallback
        H, W = sm.shape
        blocked = np.zeros((H, W), bool)
        ys, xs = [], []
        for idx in order:
            y, x = divmod(int(idx), W)
            if blocked[y, x]:
                continue
            ys.append(y), xs.append(x)
            if len(ys) >= k:
                break
            blocked[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1] = True
        ys, xs = np.array(ys, int), np.array(xs, int)
    return ys, xs, sm[ys, xs]


def normed_gradient_fast(img):
    if img.ndim == 2:
        img = img[:, :, None]
    return _ng(np.ascontiguousarray(img))


def binary_scores_fast(g, scorer):
    return _binary_scores(g, scorer.a_plus, scorer.betas.astype(np.float64), scorer.n_bits)
