"""Paired comparison of generalized-check decoders under total depolarizing noise (p/3 per Pauli).

Every decoder sees the same errors at a given (code, p). A shot fails if a decoded half does not
match its syndrome or the residual flips a logical. Decoder specs (JSON list, see the
decoders_*.json files) have a ``kind``:

- ``mbp``    : MbpLrbmsDecoder (joint X/Z decoding); any gc_method, optional OSD.
- ``gmbp``   : the hybrid GMBP4 decoder of Mostad et al. (GmbpDecoder).
- ``binary`` : one LrbmsDecoder per CSS half with the marginal rate 2p/3.
- ``lead``   : one LeadDecoder per CSS half with the marginal rate 2p/3.
- ``bposd``  : one ldpc BpOsdDecoder per CSS half with the marginal rate 2p/3.

``groups`` selects the generalized checks (see codes.CssCode.groups).

Usage:
    python compare.py --code qt250 --decoders decoders_sogrand.json --ps 0.0296,0.0392 \
        --max-shots 20000 --max-fails 50 --out results.json
"""

import argparse
import json
import multiprocessing as mp
import os
import time
import zlib

import codes
import numpy as np

from ldpc import BpOsdDecoder, LrbmsDecoder
from ldpc.lrbms_decoder import GmbpDecoder, LeadDecoder, MbpLrbmsDecoder

_code_cache = {}


def get_code(label):
    if label not in _code_cache:
        _code_cache[label] = codes.load(label)
    return _code_cache[label]


class SplitDecoder:
    """Independent binary decoders: x-part from syndrome_z (H_Z), z-part from syndrome_x (H_X)."""

    def __init__(self, x_part, z_part):
        self.x_part, self.z_part = x_part, z_part

    def decode(self, syndrome_x, syndrome_z):
        return self.x_part.decode(syndrome_z), self.z_part.decode(syndrome_x)


def _sogrand_settings(spec):
    return dict(
        sogrand_list_size=spec.get("list_size", 4),
        sogrand_threshold=spec.get("threshold", 1e-5),
        sogrand_max_queries=spec.get("max_queries", 0),
    )


def _osd(spec):
    if spec.get("osd_order") is None:
        return {}
    return dict(osd_method=spec.get("osd_method", "osd_cs"), osd_order=spec["osd_order"])


def make_decoder(spec, code, p):
    kind = spec["kind"]
    groups = spec.get("groups")
    if kind == "mbp":
        return MbpLrbmsDecoder(
            code.hx,
            code.hz,
            error_rate=p,
            x_groups=code.groups(groups, "x"),
            z_groups=code.groups(groups, "z"),
            max_iter=spec.get("max_iter", 100),
            mu=spec.get("mu", 1.0),
            alpha=spec.get("alpha", 1.0),
            lrbms_order=spec.get("t", 0),
            schedule=spec.get("schedule", "parallel"),
            gc_method=spec.get("gc_method", "map"),
            **_osd(spec),
            **_sogrand_settings(spec),
        )
    if kind == "gmbp":
        return GmbpDecoder(
            code.hx,
            code.hz,
            error_rate=p,
            x_groups=code.groups(groups, "x"),
            z_groups=code.groups(groups, "z"),
            scaling=spec.get("scaling", 1.6),
            first_stage_iter=spec.get("first_stage_iter", 6),
            max_iter=spec.get("max_iter", 6),
            osd_order=spec.get("osd_order", 1),
            gc_method=spec.get("gc_method", "map"),
            lrbms_order=spec.get("t", 0),
        )
    marginal = 2.0 * p / 3.0
    halves = []
    for pcm, basis in ((code.hz, "z"), (code.hx, "x")):
        if kind == "binary":
            halves.append(
                LrbmsDecoder(
                    pcm,
                    error_rate=marginal,
                    check_groups=code.groups(groups, basis),
                    max_iter=spec.get("max_iter", 100),
                    ms_scaling_factor=spec.get("msf", 1.0),
                    lrbms_order=spec.get("t", 0),
                    schedule=spec.get("schedule", "parallel"),
                    gc_method=spec.get("gc_method", "map"),
                    **_osd(spec),
                    **_sogrand_settings(spec),
                )
            )
        elif kind == "lead":
            halves.append(
                LeadDecoder(
                    pcm,
                    code.groups(groups or "vertex", basis),
                    error_rate=marginal,
                    scaling=spec.get("scaling", 1.0),
                    local_lsd_order=spec.get("lsd_order", 3),
                    global_osd_order=spec.get("osd_order", 3),
                    zero_views=spec.get("zero_views", "prior"),
                )
            )
        elif kind == "bposd":
            halves.append(
                BpOsdDecoder(
                    pcm,
                    error_rate=marginal,
                    max_iter=spec.get("max_iter", pcm.shape[1]),
                    bp_method=spec.get("bp_method", "minimum_sum"),
                    ms_scaling_factor=spec.get("msf", 1.0),
                    osd_method=spec.get("osd_method", "osd_cs"),
                    osd_order=spec.get("osd_order", 3),
                )
            )
        else:
            raise ValueError(kind)
    return SplitDecoder(*halves)


def sample_depolarizing(rng, n, p):
    draw = rng.random(n)
    is_x = draw < p / 3
    is_y = (draw >= p / 3) & (draw < 2 * p / 3)
    is_z = (draw >= 2 * p / 3) & (draw < p)
    return (is_x | is_y).astype(np.uint8), (is_y | is_z).astype(np.uint8)


def run_chunk(args):
    """Returns (failures, shots, seconds) for one chunk."""
    label, spec, p, shots, seed = args
    code = get_code(label)
    decoder = make_decoder(spec, code, p)
    rng = np.random.default_rng(seed)
    hx, hz, lx, lz = code.hx, code.hz, code.lx, code.lz
    fails = 0
    seconds = 0.0
    for _ in range(shots):
        error_x, error_z = sample_depolarizing(rng, code.n, p)
        syndrome_x = (hx @ error_z % 2).astype(np.uint8)
        syndrome_z = (hz @ error_x % 2).astype(np.uint8)
        start = time.perf_counter()
        decoded_x, decoded_z = decoder.decode(syndrome_x, syndrome_z)
        seconds += time.perf_counter() - start
        residual_x = (error_x + np.asarray(decoded_x)) % 2
        residual_z = (error_z + np.asarray(decoded_z)) % 2
        failed = (
            (hz @ residual_x % 2).any()
            or (hx @ residual_z % 2).any()
            or ((lz @ residual_x) % 2).any()
            or ((lx @ residual_z) % 2).any()
        )
        fails += bool(failed)
    return fails, shots, seconds


def simulate(pool, label, spec, p, max_shots, max_fails, chunk, seed):
    fails = shots = 0
    seconds, chunk_index = 0.0, 0
    workers = pool._processes
    while shots < max_shots and fails < max_fails:
        batch = [
            (label, spec, p, chunk, seed * 1_000_003 + chunk_index + i) for i in range(workers)
        ]
        for chunk_fails, chunk_shots, chunk_seconds in pool.map(run_chunk, batch):
            fails += chunk_fails
            shots += chunk_shots
            seconds += chunk_seconds
        chunk_index += workers
    return dict(fails=fails, shots=shots, ms_per_shot=1e3 * seconds / shots)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True)
    parser.add_argument("--decoders", required=True)
    parser.add_argument("--ps", required=True)
    parser.add_argument("--only", default=None, help="comma-separated decoder names to run")
    parser.add_argument("--max-shots", type=int, default=20000)
    parser.add_argument("--max-fails", type=int, default=50)
    parser.add_argument("--chunk", type=int, default=50)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--seed-salt", default="", help="changes the error samples (tuning vs test)"
    )
    args = parser.parse_args()

    decoders = json.load(open(args.decoders))
    if args.only:
        wanted = set(args.only.split(","))
        decoders = [spec for spec in decoders if spec["name"] in wanted]
    results = json.load(open(args.out)) if args.out and os.path.exists(args.out) else []
    done = {(r["code"], r["decoder"], r["p"]) for r in results}
    with mp.Pool(args.workers) as pool:
        for p in [float(value) for value in args.ps.split(",")]:
            # same seed for every decoder at a given (code, p): paired error samples
            seed = zlib.crc32(f"{args.code}|{p}{args.seed_salt}".encode())
            for spec in decoders:
                if (args.code, spec["name"], p) in done:
                    continue
                result = simulate(
                    pool,
                    args.code,
                    spec,
                    p,
                    spec.get("max_shots", args.max_shots),
                    spec.get("max_fails", args.max_fails),
                    spec.get("chunk", args.chunk),
                    seed,
                )
                result.update(code=args.code, decoder=spec["name"], p=p)
                results.append(result)
                print(
                    f"{args.code} p={p:<7} {spec['name']:>40}: "
                    f"LER={result['fails'] / result['shots']:.2e} "
                    f"({result['fails']}/{result['shots']}) {result['ms_per_shot']:.3f} ms/shot",
                    flush=True,
                )
                if args.out:
                    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
