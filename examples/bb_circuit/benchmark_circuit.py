"""Circuit-level benchmark (BB memory experiment): LRB-MS with detector groupings vs BP / BP+OSD.

The full detector history of `rounds` rounds is decoded at once with the DEM check matrix.
A shot fails if the predicted observable flips differ from the sampled ones.

Usage: python benchmark_circuit.py --decoders decoders.json --code "[[144,12,12]]" --rounds 12 \
           --ps 0.001,0.002,0.003 --out results.json
"""

import argparse
import json
import multiprocessing as mp
import os
import time
import zlib

import numpy as np

from ldpc import BpDecoder, BpOsdDecoder, LrbmsDecoder
from ldpc.lrbms_decoder import LrbmsEnsembleDecoder

from circuit_setup import CircuitProblem

_PROBLEMS = {}


def problem(code, p, rounds):
    key = (code, p, rounds)
    if key not in _PROBLEMS:
        _PROBLEMS[key] = CircuitProblem(code, p, rounds)
    return _PROBLEMS[key]


def make_decoder(spec, P):
    H, ch = P.H, P.priors
    kind = spec["kind"]
    common = dict(error_channel=ch.tolist(), max_iter=spec.get("max_iter", 100))
    if kind == "bp":
        return BpDecoder(H, bp_method=spec.get("bp_method", "minimum_sum"), ms_scaling_factor=spec.get("msf", 0.75),
                         schedule=spec.get("schedule", "serial"), **common)
    if kind == "bposd":
        return BpOsdDecoder(H, bp_method=spec.get("bp_method", "minimum_sum"), ms_scaling_factor=spec.get("msf", 0.75),
                            schedule=spec.get("schedule", "serial"), osd_method=spec.get("osd_method", "osd_cs"),
                            osd_order=spec.get("osd_order", 7), **common)
    lrb = dict(ms_scaling_factor=spec.get("msf", 0.75), lrbms_order=spec.get("t", 0),
               schedule=spec.get("schedule", "serial"), osd_method=spec.get("osd_method", "off"),
               osd_order=spec.get("osd_order", 0))
    if kind == "lrbms":
        return LrbmsDecoder(H, check_groups=P.resolve(spec.get("groups")), **lrb, **common)
    if kind == "ensemble":
        groupings = [P.resolve(g) for g in spec["groupings"]]
        return LrbmsEnsembleDecoder(H, groupings=groupings, stop=spec.get("stop", "all"),
                                    osd_members=spec.get("osd_members", "all"), **lrb, **common)
    raise ValueError(kind)


def run_chunk(args):
    code, p, rounds, spec, shots, seed = args
    P = problem(code, p, rounds)
    dec = make_decoder(spec, P)
    dets, obs = P.circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    fails = 0
    t0 = time.perf_counter()
    for i in range(shots):
        e = dec.decode(dets[i].astype(np.uint8))
        fails += bool(((P.O @ e) % 2 != obs[i]).any())
    return fails, shots, time.perf_counter() - t0


def simulate(pool, code, p, rounds, spec, max_shots, max_fails, chunk, seed):
    fails = shots = 0
    secs, k = 0.0, 0
    w = pool._processes
    while shots < max_shots and fails < max_fails:
        for f, n, t in pool.map(run_chunk, [(code, p, rounds, spec, chunk, seed * 1_000_003 + k + i)
                                           for i in range(w)]):
            fails, shots, secs = fails + f, shots + n, secs + t
        k += w
    return dict(fails=fails, shots=shots, us_per_shot=1e6 * secs / shots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="[[144,12,12]]")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--ps", default="0.001,0.002,0.003,0.004")
    ap.add_argument("--decoders", default="decoders.json")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--max-shots", type=int, default=10000)
    ap.add_argument("--max-fails", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    decoders = json.load(open(args.decoders))
    results = json.load(open(args.out)) if os.path.exists(args.out) else []
    label = f"{args.code} x{args.rounds} rounds"
    done = {(r["code"], r["decoder"], r["p"]) for r in results}
    with mp.Pool(args.workers) as pool:
        for p in [float(x) for x in args.ps.split(",")]:
            for spec in decoders:
                if (label, spec["name"], p) in done:
                    continue
                # same stim seeds for every decoder at a given p: paired samples
                seed = zlib.crc32(f"{label}|{p}".encode())
                res = simulate(pool, args.code, p, args.rounds, spec, args.max_shots, args.max_fails,
                               args.chunk, seed)
                res.update(code=label, decoder=spec["name"], p=p)
                results.append(res)
                print(f"{label} p={p:<6} {spec['name']:>34}: LER={res['fails'] / res['shots']:.2e} "
                      f"({res['fails']}/{res['shots']}) {res['us_per_shot'] / 1000:.1f} ms/shot", flush=True)
                json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
