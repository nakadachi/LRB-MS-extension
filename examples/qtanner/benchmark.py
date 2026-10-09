"""Code-capacity benchmark of Qulid vs BP / BP+OSD on quantum Tanner codes.

Noise model: i.i.d. bit flips (X errors) with probability ``p`` on every qubit.
The syndrome ``s = H_Z e`` is decoded with ``H_Z``; a shot fails if the residual
``e + e_hat`` has a non-zero syndrome or anticommutes with a Z logical.

Usage::

    python benchmark.py --codes codes.json --out results.json
"""

import argparse
import json
import multiprocessing as mp
import os
import time
import zlib

import numpy as np

from ldpc import BpDecoder, BpLsdDecoder, BpOsdDecoder, LrbmsDecoder

from qtanner_codes import build_qtanner, group_table
from search_codes import GROUPS

MAX_ITER = 100


def make_decoder(spec, code, p):
    """Builds the decoder described by a decoders.json entry for H_Z of the code."""
    hz = code.hz
    kind = spec["kind"]
    if kind == "bp":
        return BpDecoder(
            hz,
            error_rate=p,
            bp_method="minimum_sum",
            ms_scaling_factor=spec["msf"],
            schedule="serial",
            max_iter=MAX_ITER,
        )
    if kind == "bposd":
        return BpOsdDecoder(
            hz,
            error_rate=p,
            bp_method=spec.get("bp_method", "minimum_sum"),
            ms_scaling_factor=spec.get("msf", 1.0),
            schedule="serial",
            max_iter=MAX_ITER,
            osd_method=spec["osd_method"],
            osd_order=spec["osd_order"],
        )
    if kind == "bplsd":
        return BpLsdDecoder(
            hz,
            error_rate=p,
            bp_method="minimum_sum",
            ms_scaling_factor=spec["msf"],
            schedule="serial",
            max_iter=MAX_ITER,
            lsd_method="lsd_cs",
            lsd_order=spec["lsd_order"],
        )
    if kind == "lrbms":
        # "vertex": one GC per Tanner-graph vertex (its local tensor code); int: greedy overlap
        groups = code.vertex_groups("z") if spec["groups"] == "vertex" else spec["groups"]
        return LrbmsDecoder(
            hz,
            error_rate=p,
            check_groups=groups,
            max_iter=spec.get("max_iter", MAX_ITER),
            ms_scaling_factor=spec["msf"],
            lrbms_order=spec.get("t", 0),
            gc_method=spec.get("gc_method", "lrbms"),
            schedule="serial",
            osd_method=spec.get("osd_method", "off"),
            osd_order=spec.get("osd_order", 0),
        )
    raise ValueError(kind)


_code_cache = {}


def load_code(entry):
    """Builds (and caches) the code of a codes.json entry."""
    label = entry["label"]
    if label not in _code_cache:
        from search_large import GROUPS as LARGE_GROUPS
        from search_large import LOCAL

        all_groups = {**GROUPS, **LARGE_GROUPS}
        h_a, h_b, _ = LOCAL[entry.get("local", "633")]
        mul, _, _ = group_table(all_groups[entry["group"]]())
        _code_cache[label] = build_qtanner(mul, entry["A"], entry["B"], h_a, h_b, name=label)
    return _code_cache[label]


def run_chunk(args):
    """Returns (failures, unconverged, shots, seconds) for one chunk of shots."""
    entry, spec, p, shots, seed = args
    code = load_code(entry)
    decoder = make_decoder(spec, code, p)
    rng = np.random.default_rng(seed)
    hz = code.hz
    lz = code.lz.astype(np.int64)
    fails, unconverged = 0, 0
    start = time.perf_counter()
    for _ in range(shots):
        error = (rng.random(code.n) < p).astype(np.uint8)
        syndrome = (hz @ error) % 2
        decoded = decoder.decode(syndrome.astype(np.uint8))
        residual = (error + decoded) % 2
        if ((hz @ residual) % 2).any() or ((lz @ residual) % 2).any():
            fails += 1
        if hasattr(decoder, "converge") and not decoder.converge:
            unconverged += 1
    return fails, unconverged, shots, time.perf_counter() - start


def simulate(pool, entry, spec, p, max_shots, max_fails, chunk, seed):
    """Runs chunks on all workers until max_shots or max_fails is reached."""
    fails = unconverged = shots = 0
    seconds = 0.0
    chunk_index = 0
    workers = pool._processes
    while shots < max_shots and fails < max_fails:
        batch = [
            (entry, spec, p, chunk, seed * 1_000_003 + chunk_index + i) for i in range(workers)
        ]
        chunk_index += workers
        for chunk_fails, chunk_unconverged, chunk_shots, chunk_seconds in pool.map(
            run_chunk, batch
        ):
            fails += chunk_fails
            unconverged += chunk_unconverged
            shots += chunk_shots
            seconds += chunk_seconds
    return dict(
        fails=fails, unconverged=unconverged, shots=shots, us_per_shot=1e6 * seconds / shots
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", default="codes.json")
    parser.add_argument("--decoders", default="decoders.json")
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--max-shots", type=int, default=20000)
    parser.add_argument("--max-fails", type=int, default=100)
    parser.add_argument("--chunk", type=int, default=100)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    codes = json.load(open(args.codes))
    decoders = json.load(open(args.decoders))
    results = json.load(open(args.out)) if os.path.exists(args.out) else []
    done = {(r["code"], r["decoder"], r["p"]) for r in results}
    with mp.Pool(args.workers) as pool:
        for entry in codes:
            label = entry["label"]
            for p in entry["ps"]:
                for spec in decoders:
                    if (label, spec["name"], p) in done:
                        continue
                    if p in spec.get("skip_p", {}).get(label, []):
                        continue
                    only_ps = spec.get("only_p", {}).get(label)
                    if only_ps is not None and p not in only_ps:
                        continue
                    # same seed for every decoder at a given (code, p): paired error samples
                    seed = zlib.crc32(f"{label}|{p}".encode())
                    result = simulate(
                        pool,
                        entry,
                        spec,
                        p,
                        spec.get("max_shots", args.max_shots),
                        spec.get("max_fails", args.max_fails),
                        spec.get("chunk", args.chunk),
                        seed,
                    )
                    result.update(code=label, decoder=spec["name"], p=p)
                    results.append(result)
                    print(
                        f"{label:>22} p={p:<6} {spec['name']:>24}: "
                        f"LER={result['fails'] / result['shots']:.2e} "
                        f"({result['fails']}/{result['shots']}) "
                        f"unconv={result['unconverged'] / result['shots']:.2e} "
                        f"{result['us_per_shot']:.0f} us/shot",
                        flush=True,
                    )
                    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
