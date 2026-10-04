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

_problem_cache = {}


def problem(code, p, rounds):
    key = (code, p, rounds)
    if key not in _problem_cache:
        _problem_cache[key] = CircuitProblem(code, p, rounds)
    return _problem_cache[key]


def make_decoder(spec, circuit_problem):
    check_matrix = circuit_problem.check_matrix
    kind = spec["kind"]
    common = dict(error_channel=circuit_problem.priors.tolist(), max_iter=spec.get("max_iter", 100))
    if kind == "bp":
        return BpDecoder(
            check_matrix,
            bp_method=spec.get("bp_method", "minimum_sum"),
            ms_scaling_factor=spec.get("msf", 0.75),
            schedule=spec.get("schedule", "serial"),
            **common,
        )
    if kind == "bposd":
        return BpOsdDecoder(
            check_matrix,
            bp_method=spec.get("bp_method", "minimum_sum"),
            ms_scaling_factor=spec.get("msf", 0.75),
            schedule=spec.get("schedule", "serial"),
            osd_method=spec.get("osd_method", "osd_cs"),
            osd_order=spec.get("osd_order", 7),
            **common,
        )
    lrbms_settings = dict(
        ms_scaling_factor=spec.get("msf", 0.75),
        lrbms_order=spec.get("t", 0),
        schedule=spec.get("schedule", "serial"),
        osd_method=spec.get("osd_method", "off"),
        osd_order=spec.get("osd_order", 0),
    )
    if kind == "lrbms":
        return LrbmsDecoder(
            check_matrix,
            check_groups=circuit_problem.resolve(spec.get("groups")),
            **lrbms_settings,
            **common,
        )
    if kind == "ensemble":
        groupings = [circuit_problem.resolve(grouping) for grouping in spec["groupings"]]
        return LrbmsEnsembleDecoder(
            check_matrix,
            groupings=groupings,
            stop=spec.get("stop", "all"),
            osd_members=spec.get("osd_members", "all"),
            **lrbms_settings,
            **common,
        )
    raise ValueError(kind)


def run_chunk(args):
    """Returns (failures, shots, seconds) for one chunk of stim samples."""
    code, p, rounds, spec, shots, seed = args
    circuit_problem = problem(code, p, rounds)
    decoder = make_decoder(spec, circuit_problem)
    sampler = circuit_problem.circuit.compile_detector_sampler(seed=seed)
    detectors, observables = sampler.sample(shots, separate_observables=True)
    fails = 0
    start = time.perf_counter()
    for shot in range(shots):
        error = decoder.decode(detectors[shot].astype(np.uint8))
        predicted = (circuit_problem.observables @ error) % 2
        fails += bool((predicted != observables[shot]).any())
    return fails, shots, time.perf_counter() - start


def simulate(pool, code, p, rounds, spec, max_shots, max_fails, chunk, seed):
    fails = shots = 0
    seconds, chunk_index = 0.0, 0
    workers = pool._processes
    while shots < max_shots and fails < max_fails:
        batch = [
            (code, p, rounds, spec, chunk, seed * 1_000_003 + chunk_index + i)
            for i in range(workers)
        ]
        for chunk_fails, chunk_shots, chunk_seconds in pool.map(run_chunk, batch):
            fails += chunk_fails
            shots += chunk_shots
            seconds += chunk_seconds
        chunk_index += workers
    return dict(fails=fails, shots=shots, us_per_shot=1e6 * seconds / shots)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", default="[[144,12,12]]")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--ps", default="0.001,0.002,0.003,0.004")
    parser.add_argument("--decoders", default="decoders.json")
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--max-shots", type=int, default=10000)
    parser.add_argument("--max-fails", type=int, default=100)
    parser.add_argument("--chunk", type=int, default=50)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    decoders = json.load(open(args.decoders))
    results = json.load(open(args.out)) if os.path.exists(args.out) else []
    label = f"{args.code} x{args.rounds} rounds"
    done = {(r["code"], r["decoder"], r["p"]) for r in results}
    with mp.Pool(args.workers) as pool:
        for p in [float(value) for value in args.ps.split(",")]:
            for spec in decoders:
                if (label, spec["name"], p) in done:
                    continue
                # same stim seeds for every decoder at a given p: paired samples
                seed = zlib.crc32(f"{label}|{p}".encode())
                result = simulate(
                    pool,
                    args.code,
                    p,
                    args.rounds,
                    spec,
                    spec.get("max_shots", args.max_shots),
                    spec.get("max_fails", args.max_fails),
                    args.chunk,
                    seed,
                )
                result.update(code=label, decoder=spec["name"], p=p)
                results.append(result)
                print(
                    f"{label} p={p:<6} {spec['name']:>34}: "
                    f"LER={result['fails'] / result['shots']:.2e} "
                    f"({result['fails']}/{result['shots']}) "
                    f"{result['us_per_shot'] / 1000:.1f} ms/shot",
                    flush=True,
                )
                json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
