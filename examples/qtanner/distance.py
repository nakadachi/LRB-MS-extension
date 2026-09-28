"""Randomised information-set upper bound on the X/Z distance of a CSS code."""

import numpy as np
import scipy.sparse

import ldpc.mod2


def _rref_rows(mat: np.ndarray) -> np.ndarray:
    """Reduced row echelon form over GF(2) (rows are codewords, modified in place)."""
    mat = mat.copy()
    r = 0
    rows, cols = mat.shape
    for c in range(cols):
        if r == rows:
            break
        piv = np.flatnonzero(mat[r:, c])
        if piv.size == 0:
            continue
        p = r + piv[0]
        if p != r:
            mat[[r, p]] = mat[[p, r]]
        hits = np.flatnonzero(mat[:, c])
        hits = hits[hits != r]
        mat[hits] ^= mat[r]
        r += 1
    return mat[:r]


def distance_upper_bound(h_commute, logicals_dual, trials=200, seed=0):
    """Min weight of a codeword of ker(h_commute) that anticommutes with some dual logical.

    For the X-distance pass ``h_commute = hz`` and ``logicals_dual = lz``.
    """
    rng = np.random.default_rng(seed)
    ker = ldpc.mod2.kernel(scipy.sparse.csr_matrix(h_commute))
    ker = np.asarray(ker.todense() if scipy.sparse.issparse(ker) else ker, dtype=np.uint8) % 2
    lz = np.asarray(logicals_dual, dtype=np.uint8)
    n = ker.shape[1]
    best, best_word = n + 1, None
    for _ in range(trials):
        perm = rng.permutation(n)
        red = _rref_rows(ker[:, perm])
        words = np.empty_like(red)
        words[:, perm] = red
        nontrivial = ((words.astype(np.int64) @ lz.T.astype(np.int64)) % 2).any(axis=1)
        if not nontrivial.any():
            continue
        wts = words.sum(axis=1)
        wts[~nontrivial] = n + 1
        i = int(np.argmin(wts))
        if wts[i] < best:
            best, best_word = int(wts[i]), words[i]
    return best, best_word
