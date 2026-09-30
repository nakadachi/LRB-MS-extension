"""Code-capacity benchmark of LRB-MS (single and grouping ensembles) vs BP / BP+OSD on BB codes.

Usage: python benchmark_bb.py [--codes codes.json] [--decoders decoders.json] [--out results.json]
"""

import argparse
import json
import multiprocessing as mp
import os
import zlib

from harness import simulate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="codes.json")
    ap.add_argument("--decoders", default="decoders.json")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--max-shots", type=int, default=20000)
    ap.add_argument("--max-fails", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    codes, decoders = json.load(open(args.codes)), json.load(open(args.decoders))
    results = json.load(open(args.out)) if os.path.exists(args.out) else []
    done = {(r["code"], r["decoder"], r["p"]) for r in results}
    with mp.Pool(args.workers) as pool:
        for entry in codes:
            for p in entry["ps"]:
                for spec in decoders:
                    if (entry["label"], spec["name"], p) in done:
                        continue
                    # same seed for every decoder at a given (code, p): paired error samples
                    seed = zlib.crc32(f"{entry['label']}|{p}".encode())
                    res = simulate(pool, entry["label"], spec, p, args.max_shots, args.max_fails, args.chunk, seed)
                    res.update(code=entry["label"], decoder=spec["name"], p=p)
                    results.append(res)
                    print(f"{entry['label']:>14} p={p:<5} {spec['name']:>32}: "
                          f"LER={res['fails'] / res['shots']:.2e} ({res['fails']}/{res['shots']}) "
                          f"{res['us_per_shot']:.0f} us/shot", flush=True)
                    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
