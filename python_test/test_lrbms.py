import itertools

import numpy as np
import pytest
import scipy.sparse

from ldpc.mod2 import rank

from ldpc import BpDecoder
from ldpc.lrbms_decoder import (
    LrbmsDecoder,
    consecutive_check_groups,
    overlap_check_groups,
)


def random_ldpc(m, n, col_weight, seed):
    rng = np.random.default_rng(seed)
    H = np.zeros((m, n), dtype=np.uint8)
    for j in range(n):
        H[rng.choice(m, col_weight, replace=False), j] = 1
    return H


def brute_force_maxlog(H, s, llr):
    """Exact max-log extrinsic LLRs of every bit over the syndrome coset {e : He = s}."""
    m, n = H.shape
    best = np.full((n, 2), np.inf)
    for bits in itertools.product([0, 1], repeat=n):
        e = np.array(bits, dtype=np.uint8)
        if np.any(H @ e % 2 != s):
            continue
        cost = float(e @ llr)
        for k in range(n):
            ext = cost - e[k] * llr[k]
            best[k, e[k]] = min(best[k, e[k]], ext)
    out = np.empty(n)
    for k in range(n):
        if np.isfinite(best[k]).all():
            out[k] = best[k, 1] - best[k, 0]
        else:
            out[k] = np.nan  # forced bit
    return out


def bb_code_144():
    """Bivariate bicycle [[144,12,12]] code (Bravyi et al. 2024): returns H_X."""
    l, m = 12, 6
    S_l = np.roll(np.eye(l, dtype=np.uint8), 1, axis=1)
    S_m = np.roll(np.eye(m, dtype=np.uint8), 1, axis=1)
    x = np.kron(S_l, np.eye(m, dtype=np.uint8))
    y = np.kron(np.eye(l, dtype=np.uint8), S_m)
    mp = np.linalg.matrix_power
    A = (mp(x, 3) + y + mp(y, 2)) % 2
    B = (mp(y, 3) + x + mp(x, 2)) % 2
    return np.hstack([A, B]).astype(np.uint8)


# --------------------------------------------------------------------------- basic

def test_constructor_and_properties():
    H = random_ldpc(10, 20, 3, 0)
    d = LrbmsDecoder(H, error_rate=0.1, check_groups=3, lrbms_order=4,
                     ms_scaling_factor=0.8, max_iter=7, schedule="serial")
    assert d.lrbms_order == 4
    assert d.max_iter == 7
    assert d.schedule == "serial"
    assert d.gc_method == "lrbms"
    assert d.max_group_size <= 3
    assert sorted(r for g in d.check_groups for r in g) == list(range(10))
    d.gc_method = "trellis"
    assert d.gc_method == "trellis"
    with pytest.raises(ValueError):
        d.gc_method = "foo"
    with pytest.raises(ValueError):
        LrbmsDecoder(H)  # no channel
    with pytest.raises(ValueError):
        LrbmsDecoder(H, error_rate=0.1, check_groups=[[0, 1]])  # rows not covered


def test_zero_syndrome():
    H = random_ldpc(10, 20, 3, 0)
    d = LrbmsDecoder(H, error_rate=0.1, check_groups=2)
    out = d.decode(np.zeros(10, dtype=np.uint8))
    assert not out.any() and d.converge


def test_hamming_single_error():
    H = np.array([[0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8)
    for j in range(7):
        e = np.zeros(7, dtype=np.uint8)
        e[j] = 1
        for method in ["lrbms", "trellis"]:
            d = LrbmsDecoder(H, error_rate=0.05, check_groups=[[0, 1, 2]], gc_method=method)
            assert np.array_equal(d.decode(H @ e % 2), e)
            assert d.converge and d.iterations == 1


# ----------------------------------------------------- equivalence / exactness

def test_single_row_groups_equal_min_sum_bp():
    """With one row per generalized check (parallel schedule), LRB-MS is exactly min-sum BP."""
    H = random_ldpc(30, 60, 3, 1)
    p = 0.05
    rng = np.random.default_rng(2)
    bp = BpDecoder(H, error_rate=p, bp_method="minimum_sum", schedule="parallel",
                   ms_scaling_factor=0.75, max_iter=5)
    lr = LrbmsDecoder(H, error_rate=p, ms_scaling_factor=0.75, max_iter=5)
    for _ in range(200):
        e = (rng.random(60) < p).astype(np.uint8)
        s = H @ e % 2
        assert np.array_equal(bp.decode(s), lr.decode(s))
        if not lr.converge:
            np.testing.assert_allclose(bp.log_prob_ratios, lr.log_prob_ratios, atol=1e-9)


@pytest.mark.parametrize("seed", range(6))
def test_trellis_is_exact_maxlog(seed):
    """One GC holding all rows, one iteration: posterior = channel + exact max-log extrinsic."""
    rng = np.random.default_rng(seed)
    m, n = 4, 11
    H = (rng.random((m, n)) < 0.45).astype(np.uint8)
    probs = rng.uniform(0.02, 0.3, n)
    llr = np.log((1 - probs) / probs)
    e = (rng.random(n) < 0.3).astype(np.uint8)
    s = H @ e % 2
    if not s.any():
        s[0] = 1 if H[0].any() else 0
    if not s.any():
        pytest.skip("degenerate")
    d = LrbmsDecoder(H, error_channel=probs, check_groups=[list(range(m))],
                     gc_method="trellis", max_iter=1, llr_clip=1e9)
    d.decode(s)
    ref = brute_force_maxlog(H, s, llr)
    got = d.log_prob_ratios - llr
    mask = ~np.isnan(ref)
    np.testing.assert_allclose(got[mask], ref[mask], atol=1e-9)


@pytest.mark.parametrize("seed", range(6))
def test_lrbms_exact_when_list_covers_coset(seed):
    """If n - rank <= 2 and the order covers all MRB pairs, the LRB-MS list is the whole coset."""
    rng = np.random.default_rng(100 + seed)
    n = 7
    while True:
        H = (rng.random((5, n)) < 0.5).astype(np.uint8)
        if rank(H) == 5:
            break
    probs = rng.uniform(0.02, 0.3, n)
    llr = np.log((1 - probs) / probs)
    e = (rng.random(n) < 0.3).astype(np.uint8)
    e[0] = 1
    s = H @ e % 2
    d = LrbmsDecoder(H, error_channel=probs, check_groups=[list(range(5))],
                     lrbms_order=n, max_iter=1, llr_clip=1e9)
    d.decode(s)
    ref = brute_force_maxlog(H, s, llr)
    got = d.log_prob_ratios - llr
    mask = ~np.isnan(ref)
    np.testing.assert_allclose(got[mask], ref[mask], atol=1e-9)


# ------------------------------------------------------------------- helpers

def test_group_helpers():
    assert consecutive_check_groups(7, 3) == [[0, 1, 2], [3, 4, 5], [6]]
    H = bb_code_144()
    groups = overlap_check_groups(H, 4)
    flat = sorted(r for g in groups for r in g)
    assert flat == list(range(H.shape[0]))
    assert max(len(g) for g in groups) <= 4
    groups_sp = overlap_check_groups(scipy.sparse.csr_matrix(H), 4)
    assert groups == groups_sp


# ------------------------------------------------------------- performance

def test_bb144_generalized_checks_do_not_hurt():
    """Code-capacity X errors on BB [[144,12,12]]: grouped LRB-MS should match or beat min-sum."""
    H = bb_code_144()
    n = H.shape[1]
    p = 0.06
    rng = np.random.default_rng(7)
    ms = LrbmsDecoder(H, error_rate=p, max_iter=50, ms_scaling_factor=0.8)
    gc = LrbmsDecoder(H, error_rate=p, check_groups=4, lrbms_order=4, max_iter=50, ms_scaling_factor=0.8)
    fail_ms = fail_gc = 0
    for _ in range(300):
        e = (rng.random(n) < p).astype(np.uint8)
        s = H @ e % 2
        ms.decode(s)
        fail_ms += not ms.converge
        gc.decode(s)
        fail_gc += not gc.converge
    assert fail_gc <= fail_ms + 5
