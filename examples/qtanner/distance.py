"""Randomised information-set upper bound on the X/Z distance of a CSS code."""

import numpy as np
import scipy.sparse

import ldpc.mod2


def _rref_rows(mat: np.ndarray) -> np.ndarray:
    """Reduced row echelon form over GF(2), keeping only the nonzero rows."""
    mat = mat.copy()
    rank = 0
    row_count, col_count = mat.shape
    for col in range(col_count):
        if rank == row_count:
            break
        candidates = np.flatnonzero(mat[rank:, col])
        if candidates.size == 0:
            continue
        pivot = rank + candidates[0]
        if pivot != rank:
            mat[[rank, pivot]] = mat[[pivot, rank]]
        others = np.flatnonzero(mat[:, col])
        others = others[others != rank]
        mat[others] ^= mat[rank]
        rank += 1
    return mat[:rank]


def distance_upper_bound(h_commute, logicals_dual, trials=200, seed=0):
    """Min weight of a codeword of ker(h_commute) that anticommutes with some dual logical.

    For the X-distance pass ``h_commute = hz`` and ``logicals_dual = lz``.
    Each trial row-reduces the kernel basis under a random column order; every reduced row is
    a codeword of low weight on the first columns of that order.
    """
    rng = np.random.default_rng(seed)
    kernel = ldpc.mod2.kernel(scipy.sparse.csr_matrix(h_commute))
    if scipy.sparse.issparse(kernel):
        kernel = kernel.todense()
    kernel = np.asarray(kernel, dtype=np.uint8) % 2
    dual = np.asarray(logicals_dual, dtype=np.uint8)
    n = kernel.shape[1]
    best_weight, best_word = n + 1, None
    for _ in range(trials):
        column_order = rng.permutation(n)
        reduced = _rref_rows(kernel[:, column_order])
        words = np.empty_like(reduced)
        words[:, column_order] = reduced
        nontrivial = ((words.astype(np.int64) @ dual.T.astype(np.int64)) % 2).any(axis=1)
        if not nontrivial.any():
            continue
        weights = words.sum(axis=1)
        weights[~nontrivial] = n + 1
        lightest = int(np.argmin(weights))
        if weights[lightest] < best_weight:
            best_weight, best_word = int(weights[lightest]), words[lightest]
    return best_weight, best_word
