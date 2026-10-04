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


# ------------------------------------------------- MAP and SOGRAND check rules


def brute_force_map(H, s, llr):
    """Exact sum-product extrinsic LLRs of every bit over the syndrome coset {e : He = s}."""
    n = H.shape[1]
    p1 = 1.0 / (1.0 + np.exp(llr))
    p0 = 1.0 - p1
    mass = np.zeros((n, 2))
    for bits in itertools.product([0, 1], repeat=n):
        e = np.array(bits, dtype=np.uint8)
        if np.any(H @ e % 2 != s):
            continue
        prob = np.where(e == 1, p1, p0)
        for k in range(n):
            mass[k, e[k]] += np.prod(np.delete(prob, k))
    with np.errstate(divide="ignore"):
        return np.log(mass[:, 0]) - np.log(mass[:, 1])


def sogrand_reference(H, s, llr, list_size, threshold):
    """Python transcription of SOGRAND: ORBGRAND query order, not-in-list estimate."""
    n = len(llr)
    reliability = np.abs(llr)
    hard = (llr <= 0).astype(np.uint8)
    order = np.argsort(reliability, kind="stable")
    sorted_rel = reliability[order]
    half = int(np.floor(n / 2 + 0.5))
    slope = (sorted_rel[half - 1] - sorted_rel[0]) / (half - 1) if half > 1 else 0.0
    intercept = max(int(np.floor(sorted_rel[0] / slope - 1 + 0.5)), 0) if slope > 0 else 0
    patterns = [r for w in range(n + 1) for r in itertools.combinations(range(1, n + 1), w)]
    patterns.sort(key=lambda ranks: (intercept * len(ranks) + sum(ranks), len(ranks), ranks))
    pm_hard = np.sum(np.log1p(np.exp(-reliability)))
    scale = 2.0 ** -rank(H)
    not_queried, found = 1.0, []
    for ranks in patterns:
        positions = [order[r - 1] for r in ranks]
        prob = np.exp(-(pm_hard + sum(reliability[q] for q in positions)))
        not_queried -= prob
        e = hard.copy()
        e[positions] ^= 1
        if np.array_equal(H @ e % 2, s):
            found.append((e, prob))
            mass = sum(f[1] for f in found)
            confidence = mass / (mass + scale * max(not_queried, 0.0))
            if confidence > 1 - threshold or len(found) >= list_size:
                break
    eps = np.finfo(float).eps
    not_in_list = max(scale * max(not_queried, 0.0), eps)
    prior1 = np.clip(1.0 / (1.0 + np.exp(llr)), eps, 1 - eps)
    p1 = sum(prob * e for e, prob in found) + not_in_list * prior1
    p0 = sum(prob * (1 - e) for e, prob in found) + not_in_list * (1 - prior1)
    return np.log(p0) - np.log(p1) - llr


def _single_gc_extrinsic(H, s, probs, **kwargs):
    """Extrinsic output of one GC holding all rows of H (one parallel iteration)."""
    d = LrbmsDecoder(H, error_channel=probs, check_groups=[list(range(H.shape[0]))],
                     max_iter=1, llr_clip=1e9, **kwargs)
    d.decode(s)
    return d.log_prob_ratios - np.log((1 - probs) / probs)


def _random_gc(seed, m=4, n=10):
    rng = np.random.default_rng(seed)
    H = (rng.random((m, n)) < 0.45).astype(np.uint8)
    H[:, rng.integers(n)] = 1   # no all-zero rows
    for column in np.flatnonzero(~H.any(axis=0)):   # every bit inside the check's support
        H[rng.integers(m), column] = 1
    probs = rng.uniform(0.02, 0.35, n)
    e = (rng.random(n) < 0.3).astype(np.uint8)
    if not (H @ e % 2).any():   # the decoder returns early on a zero syndrome
        e[0] ^= 1
    return H, H @ e % 2, probs


@pytest.mark.parametrize("seed", range(8))
def test_map_is_exact_sum_product(seed):
    H, s, probs = _random_gc(seed)
    ref = brute_force_map(H, s, np.log((1 - probs) / probs))
    got = _single_gc_extrinsic(H, s, probs, gc_method="map")
    mask = np.isfinite(ref)
    np.testing.assert_allclose(got[mask], ref[mask], rtol=1e-9, atol=1e-9)


def test_map_single_row_groups_equal_product_sum_bp():
    """With one row per check, the MAP rule is the box-plus rule: sum-product BP."""
    H = random_ldpc(30, 60, 3, 1)
    p = 0.05
    rng = np.random.default_rng(2)
    bp = BpDecoder(H, error_rate=p, bp_method="product_sum", schedule="parallel", max_iter=5)
    lr = LrbmsDecoder(H, error_rate=p, gc_method="map", max_iter=5)
    for _ in range(200):
        e = (rng.random(60) < p).astype(np.uint8)
        s = H @ e % 2
        assert np.array_equal(bp.decode(s), lr.decode(s))
        if not lr.converge:
            np.testing.assert_allclose(bp.log_prob_ratios, lr.log_prob_ratios, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("seed", range(6))
def test_sogrand_without_list_limit_is_map(seed):
    """Querying every pattern leaves no unexplored mass: SOGRAND equals the exact MAP rule."""
    H, s, probs = _random_gc(seed, n=9)
    ref = brute_force_map(H, s, np.log((1 - probs) / probs))
    got = _single_gc_extrinsic(H, s, probs, gc_method="sogrand", sogrand_list_size=2**10,
                               sogrand_threshold=0.0)
    mask = np.isfinite(ref)
    np.testing.assert_allclose(got[mask], ref[mask], rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("seed", range(10))
def test_sogrand_matches_reference_transcription(seed):
    H, s, probs = _random_gc(100 + seed, m=4, n=10)
    llr = np.log((1 - probs) / probs)
    ref = sogrand_reference(H, s, llr, list_size=4, threshold=1e-5)
    got = _single_gc_extrinsic(H, s, probs, gc_method="sogrand")
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9)


def test_gc_method_names_round_trip():
    H = random_ldpc(6, 12, 2, 0)
    d = LrbmsDecoder(H, error_rate=0.05)
    for name in ("lrbms", "trellis", "map", "sogrand"):
        d.gc_method = name
        assert d.gc_method == name
    d.sogrand_list_size = 7
    assert d.sogrand_list_size == 7


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


# ------------------------------------------------------------- OSD fallback

def test_osd_fallback_always_satisfies_syndrome():
    H = bb_code_144()
    n = H.shape[1]
    p = 0.06
    rng = np.random.default_rng(11)
    d = LrbmsDecoder(H, error_rate=p, check_groups=9, lrbms_order=8, max_iter=20,
                     ms_scaling_factor=0.8, osd_method="osd_cs", osd_order=7)
    assert d.osd_method == "OSD_CS" and d.osd_order == 7
    used = 0
    for _ in range(200):
        e = (rng.random(n) < p).astype(np.uint8)
        s = H @ e % 2
        c = d.decode(s)
        assert not (H @ c % 2 != s).any()
        used += d.osd_used
        assert d.osd_used == (not d.converge)
        assert np.array_equal(d.decoding, c)
    assert used > 0


def test_single_row_groups_with_osd_equal_bposd():
    """ell=1 + OSD must reproduce ldpc's BpOsdDecoder bit-for-bit (same LLRs fed to the same OSD)."""
    from ldpc import BpOsdDecoder
    H = bb_code_144()
    n = H.shape[1]
    p = 0.05
    rng = np.random.default_rng(12)
    for method, order in [("osd_0", 0), ("osd_cs", 7)]:
        a = BpOsdDecoder(H, error_rate=p, bp_method="ms", schedule="parallel", ms_scaling_factor=0.8,
                         max_iter=30, osd_method=method, osd_order=order)
        b = LrbmsDecoder(H, error_rate=p, ms_scaling_factor=0.8, max_iter=30,
                         osd_method=method, osd_order=order)
        for _ in range(150):
            e = (rng.random(n) < p).astype(np.uint8)
            s = H @ e % 2
            assert np.array_equal(a.decode(s), b.decode(s))


def test_osd_invalid_args():
    H = random_ldpc(10, 20, 3, 0)
    with pytest.raises(ValueError):
        LrbmsDecoder(H, error_rate=0.1, osd_method="nope")
    d = LrbmsDecoder(H, error_rate=0.1)
    assert d.osd_method == "OSD_OFF" and not d.osd_used


# ------------------------------------------------------------- ensemble


def test_permuted_groupings_are_distinct_partitions():
    from ldpc.lrbms_decoder import permuted_overlap_groupings

    H = bb_code_144()
    groupings = permuted_overlap_groupings(H, 8, 4, seed=3)
    assert groupings[0] == overlap_check_groups(H, 8)
    for grouping in groupings:
        assert sorted(row for group in grouping for row in group) == list(range(H.shape[0]))
    assert len({tuple(map(tuple, sorted(grouping))) for grouping in groupings}) == 4


def test_ensemble_output_is_cheapest_valid_member():
    from ldpc.lrbms_decoder import LrbmsEnsembleDecoder

    H = bb_code_144()
    n = H.shape[1]
    p = 0.06
    ensemble = LrbmsEnsembleDecoder(
        H,
        error_rate=p,
        ell=8,
        num_groupings=4,
        max_iter=50,
        ms_scaling_factor=0.75,
        schedule="serial",
        osd_method="osd_cs",
        osd_order=4,
    )
    rng = np.random.default_rng(1)
    for _ in range(40):
        e = (rng.random(n) < p).astype(np.uint8)
        s = H @ e % 2
        d = ensemble.decode(s)
        assert ensemble.converge  # OSD fallback makes every member syndrome-valid
        assert np.array_equal(H @ d % 2, s)
        member_weights = [member.decode(s).sum() for member in ensemble.members]
        assert d.sum() == min(member_weights)


def test_ensemble_single_grouping_equals_lrbms():
    from ldpc.lrbms_decoder import LrbmsEnsembleDecoder

    H = bb_code_144()
    n = H.shape[1]
    settings = dict(error_rate=0.05, max_iter=50, ms_scaling_factor=0.75, schedule="serial")
    ensemble = LrbmsEnsembleDecoder(H, groupings=[6], **settings)
    reference = LrbmsDecoder(H, check_groups=6, **settings)
    rng = np.random.default_rng(2)
    for _ in range(30):
        e = (rng.random(n) < 0.05).astype(np.uint8)
        s = H @ e % 2
        assert np.array_equal(ensemble.decode(s), reference.decode(s))


def test_ensemble_invalid_args():
    from ldpc.lrbms_decoder import LrbmsEnsembleDecoder

    H = bb_code_144()
    with pytest.raises(ValueError):
        LrbmsEnsembleDecoder(H, error_rate=0.05, stop="sometimes")
    with pytest.raises(ValueError):
        LrbmsEnsembleDecoder(H, error_rate=0.05, groupings=[])
    with pytest.raises(ValueError):
        LrbmsEnsembleDecoder(H)
    with pytest.raises(ValueError):
        LrbmsEnsembleDecoder(H, error_rate=0.05, osd_members="some")


def test_ensemble_osd_on_last_member_only():
    from ldpc.lrbms_decoder import LrbmsEnsembleDecoder

    H = bb_code_144()
    ensemble = LrbmsEnsembleDecoder(
        H,
        error_rate=0.05,
        num_groupings=3,
        stop="first",
        osd_members="last",
        osd_method="osd_cs",
        osd_order=4,
        max_iter=30,
    )
    osd_methods = [member.osd_method for member in ensemble.members]
    assert osd_methods == ["OSD_OFF", "OSD_OFF", "OSD_CS"]
    rng = np.random.default_rng(4)
    for _ in range(30):
        e = (rng.random(H.shape[1]) < 0.05).astype(np.uint8)
        s = H @ e % 2
        d = ensemble.decode(s)
        assert ensemble.converge and np.array_equal(H @ d % 2, s)


def test_ensemble_escalate_mode():
    from ldpc.lrbms_decoder import LrbmsEnsembleDecoder

    H = bb_code_144()
    n = H.shape[1]
    settings = dict(
        error_rate=0.06,
        num_groupings=4,
        max_iter=30,
        ms_scaling_factor=0.75,
        osd_method="osd_cs",
        osd_order=4,
    )
    escalate = LrbmsEnsembleDecoder(H, stop="escalate", **settings)
    run_all = LrbmsEnsembleDecoder(H, stop="all", **settings)
    rng = np.random.default_rng(5)
    for _ in range(40):
        e = (rng.random(n) < 0.06).astype(np.uint8)
        s = H @ e % 2
        d = escalate.decode(s)
        assert escalate.converge and np.array_equal(H @ d % 2, s)
        first_output = escalate.members[0].decode(s)
        if escalate.members[0].converge:
            # first member converged: escalate returns its output without running the others
            assert np.array_equal(d, first_output)
        else:
            # otherwise it behaves exactly like 'all'
            assert np.array_equal(d, run_all.decode(s))


# ------------------------------------------------------------- MBP4 + LRB-MS hybrid


def _bb144_css():
    """(H_X, H_Z) of [[144,12,12]]: H_X = [A|B] and H_Z = [B^T|A^T]."""
    hx = bb_code_144()
    n = hx.shape[1]
    A, B = hx[:, : n // 2], hx[:, n // 2 :]
    hz = np.hstack([B.T, A.T]).astype(np.uint8)
    return hx.astype(np.uint8), hz


def _sample_depolarizing(rng, n, p):
    """(x-part, z-part) of a depolarizing error with p/3 per Pauli."""
    draw = rng.random(n)
    is_x = draw < p / 3
    is_y = (draw >= p / 3) & (draw < 2 * p / 3)
    is_z = (draw >= 2 * p / 3) & (draw < p)
    return (is_x | is_y).astype(np.uint8), (is_y | is_z).astype(np.uint8)


def test_mbp_lrbms_reduces_to_binary_lrbms_for_pure_x_noise():
    from ldpc.lrbms_decoder import MbpLrbmsDecoder

    hx, hz = _bb144_css()
    n = hx.shape[1]
    p = 0.05
    groups = overlap_check_groups(hz, 6)
    hybrid = MbpLrbmsDecoder(
        hx,
        hz,
        channel=(p, 0.0, 0.0),
        z_groups=groups,
        x_groups=6,
        max_iter=50,
        mu=0.75,
        alpha=1.0,
        lrbms_order=4,
        schedule="serial",
    )
    reference = LrbmsDecoder(
        hz,
        error_rate=p,
        check_groups=groups,
        max_iter=50,
        ms_scaling_factor=0.75,
        lrbms_order=4,
        schedule="serial",
    )
    rng = np.random.default_rng(3)
    for _ in range(60):
        error_x = (rng.random(n) < p).astype(np.uint8)
        syndrome_z = hz @ error_x % 2
        decoded_x, decoded_z = hybrid.decode(np.zeros(hx.shape[0], dtype=np.uint8), syndrome_z)
        assert np.array_equal(decoded_x, reference.decode(syndrome_z))
        assert not decoded_z.any()
        assert hybrid.converge == reference.converge
        if reference.converge:
            assert hybrid.iterations == reference.iterations


def test_mbp_lrbms_depolarizing_outputs_and_setters():
    from ldpc.lrbms_decoder import MbpLrbmsDecoder

    hx, hz = _bb144_css()
    n = hx.shape[1]
    p = 0.06
    decoder = MbpLrbmsDecoder(
        hx, hz, error_rate=p, x_groups=6, z_groups=6, max_iter=60, mu=0.75, alpha=0.9, lrbms_order=1
    )
    rng = np.random.default_rng(4)
    converged = 0
    for _ in range(40):
        error_x, error_z = _sample_depolarizing(rng, n, p)
        syndrome_x, syndrome_z = hx @ error_z % 2, hz @ error_x % 2
        decoded_x, decoded_z = decoder.decode(syndrome_x, syndrome_z)
        if decoder.converge:
            converged += 1
            assert np.array_equal(hz @ decoded_x % 2, syndrome_z)
            assert np.array_equal(hx @ decoded_z % 2, syndrome_x)
    assert converged >= 30
    decoder.mu, decoder.alpha, decoder.schedule, decoder.max_iter = 0.9, 0.7, "parallel", 20
    assert (decoder.mu, decoder.alpha, decoder.schedule, decoder.max_iter) == (
        0.9,
        0.7,
        "parallel",
        20,
    )
    with pytest.raises(ValueError):
        decoder.alpha = 0.0
    with pytest.raises(ValueError):
        MbpLrbmsDecoder(hx, hx, error_rate=0.05)  # hx with itself does not commute


def test_mbp_osd_makes_every_output_syndrome_valid():
    from ldpc.lrbms_decoder import MbpLrbmsDecoder

    hx, hz = _bb144_css()
    n = hx.shape[1]
    p = 0.09
    decoder = MbpLrbmsDecoder(
        hx, hz, error_rate=p, x_groups=8, z_groups=8, max_iter=6, mu=0.625, alpha=1.0,
        schedule="parallel", gc_method="map", osd_method="osd_cs", osd_order=1,
    )
    assert decoder.gc_method == "map"
    rng = np.random.default_rng(6)
    used = 0
    for _ in range(40):
        error_x, error_z = _sample_depolarizing(rng, n, p)
        syndrome_x, syndrome_z = hx @ error_z % 2, hz @ error_x % 2
        decoded_x, decoded_z = decoder.decode(syndrome_x, syndrome_z)
        assert np.array_equal(hz @ decoded_x % 2, syndrome_z)
        assert np.array_equal(hx @ decoded_z % 2, syndrome_x)
        used += decoder.osd_used
    assert used > 0
    decoder.gc_method = "sogrand"
    assert decoder.gc_method == "sogrand"

