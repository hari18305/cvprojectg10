"""Binarized approximation of the BING filter and features (paper Sec. 3.3).

* The learned 64-D filter w is approximated by Nw binary basis vectors
      w ~= sum_j beta_j a_j ,  a_j in {-1, +1}^64            (Algorithm 1)
* An 8-bit NG feature g is approximated by its Ng most significant bit planes
      g ~= sum_k 2^(8-k) b_k ,  b_k in {0, 1}^64
* Each a_j and each b_k is packed into a single 64-bit integer, so the score
      <w, g> ~= sum_j beta_j sum_k 2^(8-k) (2 <a_j+, b_k> - |b_k|)
  needs only BITWISE AND and POPCOUNT operations (Eq. 5 in the paper).
"""
import numpy as np

from .features import FEAT

_W = np.uint64(1)


def approximate_filter(w, n_basis=2):
    """Greedy binary decomposition of a real-valued filter (Algorithm 1).

    Returns (betas, bases) where bases is an (Nw, 64) array of +/-1 values.
    """
    e = np.asarray(w, np.float64).copy()
    betas, bases = [], []
    for _ in range(n_basis):
        a = np.where(e >= 0, 1.0, -1.0)
        beta = float(a @ e) / a.size
        e -= beta * a
        betas.append(beta)
        bases.append(a)
    return np.array(betas), np.array(bases)


def pack64(bits):
    """Pack an (..., 64) array of 0/1 values into uint64 (row-major 8x8)."""
    bits = np.asarray(bits).astype(np.uint64)
    shifts = np.arange(63, -1, -1, dtype=np.uint64)
    return (bits << shifts).sum(axis=-1).astype(np.uint64)


def _window_words(plane):
    """For a binary map (H x W, values 0/1) return the packed 64-bit word of
    every 8x8 window, shape (H-7, W-7).  Implemented with shifts/ORs over rows
    as in the paper's Algorithm 2 (row bytes are built first, then 8 row bytes
    are concatenated into one 64-bit integer)."""
    H, W = plane.shape
    p = plane.astype(np.uint8)
    # r[y, x] = 8 horizontal bits starting at x, packed into one byte
    r = np.zeros((H, W - FEAT + 1), np.uint64)
    for i in range(FEAT):
        r |= p[:, i:W - FEAT + 1 + i].astype(np.uint64) << np.uint64(FEAT - 1 - i)
    # 8 consecutive row bytes -> one 64-bit word
    out = np.zeros((H - FEAT + 1, W - FEAT + 1), np.uint64)
    for j in range(FEAT):
        out |= r[j:H - FEAT + 1 + j] << np.uint64(8 * (FEAT - 1 - j))
    return out


class BinaryScorer:
    """Scores every 8x8 window of an NG map using only bit operations."""

    def __init__(self, w, n_basis=2, n_bits=4):
        self.betas, bases = approximate_filter(w, n_basis)
        self.a_plus = pack64(bases > 0)  # (Nw,) uint64, +1 positions
        self.n_bits = n_bits
        self.w_approx = (self.betas[:, None] * bases).sum(0)

    def score_map(self, ng):
        """ng: uint8 NG map -> float32 score map of shape (H-7, W-7)."""
        total = None
        for k in range(self.n_bits):
            plane = (ng >> (7 - k)) & 1
            words = _window_words(plane)
            nb = np.bitwise_count(words).astype(np.int32)  # |b_k|
            weight = float(1 << (7 - k))
            acc = None
            for beta, ap in zip(self.betas, self.a_plus):
                c = 2 * np.bitwise_count(words & ap).astype(np.int32) - nb
                acc = beta * c if acc is None else acc + beta * c
            acc = weight * acc
            total = acc if total is None else total + acc
        return total.astype(np.float32)
