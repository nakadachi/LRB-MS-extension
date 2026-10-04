"""Choose relay-ladder legs for MBP4 + LRB-MS on tuning samples (seed salt "tune").

The base leg decodes every shot. Its non-converged shots are retried with every (mu, alpha) on
a grid; legs are added greedily by (rescues - new wrong convergences) among the shots still
unconverged. Prints the failure breakdown of the base leg and the chosen ladder as JSON legs.

Usage: python tune_ladder.py --code qt432 --p 0.10 --mu 0.75 --shots 8000
"""

import argparse
import json
import multiprocessing as mp
import zlib

import numpy as np
from compare import get_code, make_decoder, sample_depolarizing

GRID_MU = (0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)
GRID_ALPHA = (0.8, 0.9, 1.0, 1.1, 1.2)


def base_spec(mu, max_iter):
    return {
        "kind": "mbp",
        "groups": "vertex",
        "gc_method": "lrbms",
        "t": 8,
        "mu": mu,
        "alpha": 1.0,
        "schedule": "serial",
        "max_iter": max_iter,
    }


def outcome(decoder, code, error_x, error_z):
    syndrome_x = (code.hx @ error_z % 2).astype(np.uint8)
    syndrome_z = (code.hz @ error_x % 2).astype(np.uint8)
    decoded_x, decoded_z = decoder.decode(syndrome_x, syndrome_z)
    if not decoder.converge:
        return "unconv"
    residual_x, residual_z = (error_x + decoded_x) % 2, (error_z + decoded_z) % 2
    wrong = ((code.lz @ residual_x) % 2).any() or ((code.lx @ residual_z) % 2).any()
    return "wrong" if wrong else "ok"


def base_chunk(args):
    label, p, mu, max_iter, shots, seed = args
    code = get_code(label)
    decoder = make_decoder(base_spec(mu, max_iter), code, p)
    rng = np.random.default_rng(seed)
    failures = []
    for _ in range(shots):
        error_x, error_z = sample_depolarizing(rng, code.n, p)
        result = outcome(decoder, code, error_x, error_z)
        if result != "ok":
            failures.append((result, error_x, error_z))
    return failures


def grid_chunk(args):
    label, p, max_iter, errors = args
    code = get_code(label)
    decoder = make_decoder(base_spec(0.75, max_iter), code, p)
    rows = []
    for error_x, error_z in errors:
        row = []
        for mu in GRID_MU:
            for alpha in GRID_ALPHA:
                decoder.mu, decoder.alpha = mu, alpha
                row.append(outcome(decoder, code, error_x, error_z))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True)
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--mu", type=float, required=True)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--shots", type=int, default=8000)
    parser.add_argument("--legs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    seed = zlib.crc32(f"{args.code}|{args.p}tune-ladder".encode())
    w = args.workers
    with mp.Pool(w) as pool:
        chunks = [
            (args.code, args.p, args.mu, args.max_iter, args.shots // w, seed + k) for k in range(w)
        ]
        failures = [f for part in pool.map(base_chunk, chunks) for f in part]
        unconverged = [(f[1], f[2]) for f in failures if f[0] == "unconv"]
        parts = [unconverged[k::w] for k in range(w)]
        tables = pool.map(grid_chunk, [(args.code, args.p, args.max_iter, part) for part in parts])
    table = [None] * len(unconverged)
    for k, rows in enumerate(tables):
        for j, row in enumerate(rows):
            table[k + j * w] = row
    grid = [(mu, alpha) for mu in GRID_MU for alpha in GRID_ALPHA]
    wrong = sum(f[0] == "wrong" for f in failures)
    print(
        f"{args.code} p={args.p}: {args.shots} shots, base (mu={args.mu}) failures "
        f"{len(failures)}: {len(unconverged)} unconverged, {wrong} converged wrong"
    )
    ladder, remaining = [[args.mu, 1.0]], list(range(len(table)))
    for _ in range(args.legs):

        def score(g):
            return sum(table[i][g] == "ok" for i in remaining) - sum(
                table[i][g] == "wrong" for i in remaining
            )

        best = max(range(len(grid)), key=score)
        if score(best) <= 0:
            break
        rescued = sum(table[i][best] == "ok" for i in remaining)
        new_wrong = sum(table[i][best] == "wrong" for i in remaining)
        ladder.append(list(grid[best]))
        print(
            f"  + leg {grid[best]}: rescues {rescued}, new wrong {new_wrong}, "
            f"left unconverged {sum(table[i][best] == 'unconv' for i in remaining)}"
        )
        remaining = [i for i in remaining if table[i][best] == "unconv"]
    print("legs:", json.dumps(ladder))


if __name__ == "__main__":
    main()
