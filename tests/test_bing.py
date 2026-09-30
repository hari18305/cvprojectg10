"""Sanity checks:  python -m pytest tests -q"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bing.binary import BinaryScorer, approximate_filter, pack64  # noqa: E402
from bing.fast import HAVE_NUMBA  # noqa: E402
from bing.features import normed_gradient  # noqa: E402
from bing.metrics import iou_matrix  # noqa: E402

rng = np.random.default_rng(0)


def test_filter_approximation_error_decreases():
    w = rng.normal(size=64)
    errs = []
    for n in range(1, 6):
        b, a = approximate_filter(w, n)
        errs.append(np.linalg.norm(w - (b[:, None] * a).sum(0)))
    assert all(e1 > e2 for e1, e2 in zip(errs, errs[1:]))


def test_binary_score_equals_bitplane_dot_product():
    """Bitwise AND+POPCOUNT score == <w_approx, g_approx> computed with floats."""
    w = rng.normal(size=64)
    ng = rng.integers(0, 256, (20, 25)).astype(np.uint8)
    sc = BinaryScorer(w, n_basis=3, n_bits=4)
    fast = sc.score_map(ng)
    g_approx = (ng >> 4) << 4  # keep top 4 bits
    for y in range(ng.shape[0] - 7):
        for x in range(ng.shape[1] - 7):
            ref = sc.w_approx @ g_approx[y:y + 8, x:x + 8].reshape(-1).astype(float)
            assert abs(ref - fast[y, x]) < 1e-3 * max(1, abs(ref))


def test_numba_matches_numpy():
    if not HAVE_NUMBA:
        return
    from bing.fast import binary_scores_fast, normed_gradient_fast
    img = rng.integers(0, 256, (40, 50, 3)).astype(np.uint8)
    assert (normed_gradient(img) == normed_gradient_fast(img)).all()
    sc = BinaryScorer(rng.normal(size=64), 2, 4)
    g = normed_gradient(img)
    assert np.allclose(sc.score_map(g), binary_scores_fast(g, sc), atol=1e-3)


def test_pack64_roundtrip():
    bits = rng.integers(0, 2, 64)
    v = int(pack64(bits))
    assert [(v >> (63 - i)) & 1 for i in range(64)] == bits.tolist()


def test_iou():
    assert np.isclose(iou_matrix([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0, 0], 1 / 3)
