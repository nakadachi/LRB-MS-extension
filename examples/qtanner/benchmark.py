"""Code-capacity benchmark of LRB-MS vs BP / BP+OSD on quantum Tanner codes.

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

from ldpc import BpDecoder, BpOsdDecoder, BpLsdDecoder, LrbmsDecoder

from qtanner_codes import build_qtanner, group_table
from search_codes import GROUPS, H_A, H_B

MAX_ITER = 100


def make_decoder(spec, code, p):
    h = code.hz
    kind = spec["kind"]
    if kind == "bp":
        return BpDecoder(h, error_rate=p, bp_method="minimum_sum", ms_scaling_factor=spec["msf"],
                         schedule="serial", max_iter=MAX_ITER)
    if kind == "bposd":
        return BpOsdDecoder(h, error_rate=p, bp_method=spec.get("bp_method", "minimum_sum"),
                            ms_scaling_factor=spec.get("msf", 1.0),
                            schedule="serial", max_iter=MAX_ITER,
                            osd_method=spec["osd_method"], osd_order=spec["osd_order"])
    if kind == "bplsd":
        return BpLsdDecoder(h, error_rate=p, bp_method="minimum_sum", ms_scaling_factor=spec["msf"],
                            schedule="serial", max_iter=MAX_ITER,
                            lsd_method="lsd_cs", lsd_order=spec["lsd_order"])
    if kind == "lrbms":
        # "vertex": one GC per Tanner-graph vertex (its local tensor code); int: greedy overlap
        groups = code.vertex_groups("z") if spec["groups"] == "vertex" else spec["groups"]
        return LrbmsDecoder(h, error_rate=p, check_groups=groups, max_iter=spec.get("max_iter", MAX_ITER),
                            ms_scaling_factor=spec["msf"], lrbms_order=spec.get("t", 0),
                            gc_method=spec.get("gc_method", "lrbms"), schedule="serial",
                            osd_method=spec.get("osd_method", "off"), osd_order=spec.get("osd_order", 0))
    raise ValueError(kind)


_CODES = {}


def load_code(entry):
    key = entry["label"]
    if key not in _CODES:
        from search_large import GROUPS as LARGE_GROUPS, LOCAL
        groups = {**GROUPS, **LARGE_GROUPS}
        h_a, h_b, _ = LOCAL[entry.get("local", "633")]
        mul, _, _ = group_table(groups[entry["group"]]())
        _CODES[key] = build_qtanner(mul, entry["A"], entry["B"], h_a, h_b, name=key)
    return _CODES[key]


def run_chunk(args):
    entry, spec, p, shots, seed = args
    code = load_code(entry)
    dec = make_decoder(spec, code, p)
    rng = np.random.default_rng(seed)
    hz = code.hz
    lz = code.lz.astype(np.int64)
    fails, unconverged = 0, 0
    t0 = time.perf_counter()
    for _ in range(shots):
        e = (rng.random(code.n) < p).astype(np.uint8)
        s = (hz @ e) % 2
        e_hat = dec.decode(s.astype(np.uint8))
        r = (e + e_hat) % 2
        if ((hz @ r) % 2).any() or ((lz @ r) % 2).any():
            fails += 1
        if hasattr(dec, "converge") and not dec.converge:
            unconverged += 1
    return fails, unconverged, shots, time.perf_counter() - t0


def simulate(pool, entry, spec, p, max_shots, max_fails, chunk, seed):
    fails = unconv = shots = 0
    secs = 0.0
    k = 0
    workers = pool._processes
    while shots < max_shots and fails < max_fails:
        batch = [(entry, spec, p, chunk, seed * 1_000_003 + k + i) for i in range(workers)]
        k += workers
        for f, u, n, t in pool.map(run_chunk, batch):
            fails += f
            unconv += u
            shots += n
            secs += t
    return dict(fails=fails, unconverged=unconv, shots=shots, us_per_shot=1e6 * secs / shots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="codes.json")
    ap.add_argument("--decoders", default="decoders.json")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--max-shots", type=int, default=20000)
    ap.add_argument("--max-fails", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    codes = json.load(open(args.codes))
    decoders = json.load(open(args.decoders))
    results = json.load(open(args.out)) if os.path.exists(args.out) else []
    done = {(r["code"], r["decoder"], r["p"]) for r in results}
    with mp.Pool(args.workers) as pool:
        for entry in codes:
            for p in entry["ps"]:
                for spec in decoders:
                    if (entry["label"], spec["name"], p) in done:
                        continue
                    if p in spec.get("skip_p", {}).get(entry["label"], []):
                        continue
                    only = spec.get("only_p", {}).get(entry["label"])
                    if only is not None and p not in only:
                        continue
                    # same seed for every decoder at a given (code, p): paired error samples
                    seed = zlib.crc32(f"{entry['label']}|{p}".encode())
                    res = simulate(pool, entry, spec, p, spec.get("max_shots", args.max_shots),
                                   spec.get("max_fails", args.max_fails), spec.get("chunk", args.chunk), seed)
                    res.update(code=entry["label"], decoder=spec["name"], p=p)
                    results.append(res)
                    print(f"{entry['label']:>22} p={p:<6} {spec['name']:>24}: "
                          f"LER={res['fails'] / res['shots']:.2e} ({res['fails']}/{res['shots']}) "
                          f"unconv={res['unconverged'] / res['shots']:.2e} "
                          f"{res['us_per_shot']:.0f} us/shot", flush=True)
                    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
