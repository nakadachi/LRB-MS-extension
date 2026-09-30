"""Paired code-capacity simulation harness for BB codes (X errors decoded with H_Z)."""

import multiprocessing as mp
import time

import numpy as np

from ldpc import BpDecoder, BpOsdDecoder, LrbmsDecoder
from ldpc.lrbms_decoder import LrbmsEnsembleDecoder, permuted_overlap_groupings

from bb_codes import BBCode

_CODES = {}


def get_code(name):
    if name not in _CODES:
        _CODES[name] = BBCode(name)
    return _CODES[name]


def resolve_groups(code, groups):
    if groups is None or isinstance(groups, int):
        return groups
    if isinstance(groups, (tuple, list)) and groups and groups[0] == "union":
        return [g for part in groups[1:] for g in resolve_groups(code, part)]
    if isinstance(groups, (tuple, list)) and groups and groups[0] == "coset":
        from bb_codes import coset_groups
        return coset_groups(code, *groups[1:])
    return groups


def make_decoder(spec, code, p):
    h, kind = code.hz, spec["kind"]
    common = dict(error_rate=p, max_iter=spec.get("max_iter", 100))
    if kind == "bp":
        return BpDecoder(h, bp_method=spec.get("bp_method", "minimum_sum"),
                         ms_scaling_factor=spec.get("msf", 0.75), schedule=spec.get("schedule", "serial"),
                         **common)
    if kind == "bposd":
        return BpOsdDecoder(h, bp_method=spec.get("bp_method", "minimum_sum"),
                            ms_scaling_factor=spec.get("msf", 0.75), schedule=spec.get("schedule", "serial"),
                            osd_method=spec.get("osd_method", "osd_cs"), osd_order=spec.get("osd_order", 7),
                            **common)
    if kind == "lrbms":
        return LrbmsDecoder(h, check_groups=resolve_groups(code, spec.get("groups")),
                            ms_scaling_factor=spec.get("msf", 0.75), lrbms_order=spec.get("t", 0),
                            gc_method=spec.get("gc_method", "lrbms"), schedule=spec.get("schedule", "serial"),
                            osd_method=spec.get("osd_method", "off"), osd_order=spec.get("osd_order", 0),
                            **common)
    if kind == "ensemble":
        groupings = [resolve_groups(code, g) for g in spec.get("extra", [])]
        groupings += permuted_overlap_groupings(code.hz, spec["ell"], spec["count"]) if spec["count"] else []
        return LrbmsEnsembleDecoder(code.hz, groupings=groupings, stop=spec.get("stop", "all"),
                                    osd_members=spec.get("osd_members", "all"),
                                    ms_scaling_factor=spec.get("msf", 0.75), lrbms_order=spec.get("t", 0),
                                    schedule="serial", osd_method=spec.get("osd_method", "off"),
                                    osd_order=spec.get("osd_order", 0), **common)
    raise ValueError(kind)


def run_chunk(args):
    code_name, spec, p, shots, seed = args
    code = get_code(code_name)
    dec = make_decoder(spec, code, p)
    rng = np.random.default_rng(seed)
    hz, lz = code.hz, code.lz.astype(np.int64)
    fails = unconv = 0
    t0 = time.perf_counter()
    for _ in range(shots):
        e = (rng.random(code.n) < p).astype(np.uint8)
        s = (hz @ e % 2).astype(np.uint8)
        r = (e + dec.decode(s)) % 2
        fails += bool(((hz @ r) % 2).any() or ((lz @ r) % 2).any())
        unconv += not getattr(dec, "converge", True)
    return fails, unconv, shots, time.perf_counter() - t0


def simulate(pool, code_name, spec, p, max_shots, max_fails, chunk, seed):
    """All decoders given the same seed see the same error samples."""
    fails = unconv = shots = 0
    secs, k = 0.0, 0
    w = pool._processes
    while shots < max_shots and fails < max_fails:
        for f, u, n, t in pool.map(run_chunk, [(code_name, spec, p, chunk, seed * 1_000_003 + k + i)
                                              for i in range(w)]):
            fails, unconv, shots, secs = fails + f, unconv + u, shots + n, secs + t
        k += w
    return dict(fails=fails, unconverged=unconv, shots=shots, us_per_shot=1e6 * secs / shots)
