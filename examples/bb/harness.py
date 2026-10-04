"""Paired code-capacity simulation harness for BB codes (X errors decoded with H_Z).

Decoder specs are dicts with a ``kind`` ('bp', 'bposd', 'lrbms' or 'ensemble') and optional
settings. Check groupings (``groups``) can be None, an int ell (greedy overlap groups), an
explicit list of row lists, ``("coset", step_x, size_x, step_y, size_y)`` or
``("union", grouping, grouping, ...)``.
"""

import time

import numpy as np

from ldpc import BpDecoder, BpOsdDecoder, LrbmsDecoder
from ldpc.lrbms_decoder import LrbmsEnsembleDecoder, permuted_overlap_groupings

from bb_codes import BBCode, coset_groups

_code_cache = {}


def get_code(name):
    if name not in _code_cache:
        _code_cache[name] = BBCode(name)
    return _code_cache[name]


def resolve_groups(code, groups):
    """Turns a grouping spec into something LrbmsDecoder accepts as check_groups."""
    if groups is None or isinstance(groups, int):
        return groups
    is_tagged = isinstance(groups, (tuple, list)) and groups
    if is_tagged and groups[0] == "union":
        return [group for part in groups[1:] for group in resolve_groups(code, part)]
    if is_tagged and groups[0] == "coset":
        return coset_groups(code, *groups[1:])
    return groups


def make_decoder(spec, code, p):
    kind = spec["kind"]
    common = dict(error_rate=p, max_iter=spec.get("max_iter", 100))
    if kind == "bp":
        return BpDecoder(
            code.hz,
            bp_method=spec.get("bp_method", "minimum_sum"),
            ms_scaling_factor=spec.get("msf", 0.75),
            schedule=spec.get("schedule", "serial"),
            **common,
        )
    if kind == "bposd":
        return BpOsdDecoder(
            code.hz,
            bp_method=spec.get("bp_method", "minimum_sum"),
            ms_scaling_factor=spec.get("msf", 0.75),
            schedule=spec.get("schedule", "serial"),
            osd_method=spec.get("osd_method", "osd_cs"),
            osd_order=spec.get("osd_order", 7),
            **common,
        )
    if kind == "lrbms":
        return LrbmsDecoder(
            code.hz,
            check_groups=resolve_groups(code, spec.get("groups")),
            ms_scaling_factor=spec.get("msf", 0.75),
            lrbms_order=spec.get("t", 0),
            gc_method=spec.get("gc_method", "lrbms"),
            schedule=spec.get("schedule", "serial"),
            osd_method=spec.get("osd_method", "off"),
            osd_order=spec.get("osd_order", 0),
            **common,
        )
    if kind == "ensemble":
        groupings = [resolve_groups(code, extra) for extra in spec.get("extra", [])]
        if spec["count"]:
            groupings += permuted_overlap_groupings(code.hz, spec["ell"], spec["count"])
        return LrbmsEnsembleDecoder(
            code.hz,
            groupings=groupings,
            stop=spec.get("stop", "all"),
            osd_members=spec.get("osd_members", "all"),
            ms_scaling_factor=spec.get("msf", 0.75),
            lrbms_order=spec.get("t", 0),
            schedule="serial",
            osd_method=spec.get("osd_method", "off"),
            osd_order=spec.get("osd_order", 0),
            **common,
        )
    raise ValueError(kind)


def run_chunk(args):
    """Returns (failures, unconverged, shots, seconds) for one chunk of shots."""
    code_name, spec, p, shots, seed = args
    code = get_code(code_name)
    decoder = make_decoder(spec, code, p)
    rng = np.random.default_rng(seed)
    hz, lz = code.hz, code.lz.astype(np.int64)
    fails = unconverged = 0
    start = time.perf_counter()
    for _ in range(shots):
        error = (rng.random(code.n) < p).astype(np.uint8)
        syndrome = (hz @ error % 2).astype(np.uint8)
        residual = (error + decoder.decode(syndrome)) % 2
        fails += bool(((hz @ residual) % 2).any() or ((lz @ residual) % 2).any())
        unconverged += not getattr(decoder, "converge", True)
    return fails, unconverged, shots, time.perf_counter() - start


def simulate(pool, code_name, spec, p, max_shots, max_fails, chunk, seed):
    """All decoders given the same seed see the same error samples."""
    fails = unconverged = shots = 0
    seconds, chunk_index = 0.0, 0
    workers = pool._processes
    while shots < max_shots and fails < max_fails:
        batch = [
            (code_name, spec, p, chunk, seed * 1_000_003 + chunk_index + i) for i in range(workers)
        ]
        for chunk_fails, chunk_unconverged, chunk_shots, chunk_seconds in pool.map(
            run_chunk, batch
        ):
            fails += chunk_fails
            unconverged += chunk_unconverged
            shots += chunk_shots
            seconds += chunk_seconds
        chunk_index += workers
    return dict(
        fails=fails, unconverged=unconverged, shots=shots, us_per_shot=1e6 * seconds / shots
    )
