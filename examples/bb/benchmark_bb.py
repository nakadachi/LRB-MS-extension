"""Code-capacity benchmark of Qulid (single and grouping ensembles) vs BP / BP+OSD on BB codes.

Usage: python benchmark_bb.py [--codes codes.json] [--decoders decoders.json] [--out results.json]
"""

import argparse
import json
import multiprocessing as mp
import os
import zlib

from harness import simulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", default="codes.json")
    parser.add_argument("--decoders", default="decoders.json")
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--max-shots", type=int, default=20000)
    parser.add_argument("--max-fails", type=int, default=100)
    parser.add_argument("--chunk", type=int, default=250)
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
                    # same seed for every decoder at a given (code, p): paired error samples
                    seed = zlib.crc32(f"{label}|{p}".encode())
                    result = simulate(
                        pool, label, spec, p, args.max_shots, args.max_fails, args.chunk, seed
                    )
                    result.update(code=label, decoder=spec["name"], p=p)
                    results.append(result)
                    print(
                        f"{label:>14} p={p:<5} {spec['name']:>32}: "
                        f"LER={result['fails'] / result['shots']:.2e} "
                        f"({result['fails']}/{result['shots']}) "
                        f"{result['us_per_shot']:.0f} us/shot",
                        flush=True,
                    )
                    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
