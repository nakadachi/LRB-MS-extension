"""Error-floor probe: collect the failures of one decoder at low p, classify them, and redecode
the same errors with other decoders.

For every failure it records whether the decoder converged (to the wrong coset) or not, the
Pauli weight of the true error and of the output, and whether the output is lighter than the
true error (then even a minimum-weight decoder would fail on it).

Usage: python floor_probe.py --code qt250 --p 0.0224 --shots 4000000 \
           --decoder decoders_floor.json:"MBP4+Qulid-8" --others decoders_floor.json
"""

import argparse
import json
import multiprocessing as mp
import zlib

import numpy as np
from compare import get_code, make_decoder, sample_depolarizing


def pauli_weight(x_part, z_part):
    return int(np.count_nonzero(np.asarray(x_part) | np.asarray(z_part)))


def classify(decoder, code, error_x, error_z):
    syndrome_x = (code.hx @ error_z % 2).astype(np.uint8)
    syndrome_z = (code.hz @ error_x % 2).astype(np.uint8)
    decoded_x, decoded_z = decoder.decode(syndrome_x, syndrome_z)
    decoded_x, decoded_z = np.asarray(decoded_x), np.asarray(decoded_z)
    valid = np.array_equal(code.hz @ decoded_x % 2, syndrome_z) and np.array_equal(
        code.hx @ decoded_z % 2, syndrome_x
    )
    if not valid:
        return "unconv", decoded_x, decoded_z
    residual_x, residual_z = (error_x + decoded_x) % 2, (error_z + decoded_z) % 2
    wrong = ((code.lz @ residual_x) % 2).any() or ((code.lx @ residual_z) % 2).any()
    return ("wrong" if wrong else "ok"), decoded_x, decoded_z


def collect(args):
    label, spec, p, shots, seed = args
    code = get_code(label)
    decoder = make_decoder(spec, code, p)
    rng = np.random.default_rng(seed)
    failures = []
    for _ in range(shots):
        error_x, error_z = sample_depolarizing(rng, code.n, p)
        result, decoded_x, decoded_z = classify(decoder, code, error_x, error_z)
        if result != "ok":
            failures.append(
                dict(
                    result=result,
                    error_weight=pauli_weight(error_x, error_z),
                    output_weight=pauli_weight(decoded_x, decoded_z),
                    error_x=error_x.tolist(),
                    error_z=error_z.tolist(),
                )
            )
    return failures


def redecode(args):
    label, spec, p, failures = args
    code = get_code(label)
    decoder = make_decoder(spec, code, p)
    return [
        classify(decoder, code, np.array(f["error_x"]), np.array(f["error_z"]))[0] for f in failures
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True)
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--decoders", required=True, help="JSON list; the first is probed")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=50000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    specs = json.load(open(args.decoders))
    probed, others = specs[0], specs[1:]
    seed = zlib.crc32(f"{args.code}|{args.p}floor".encode())
    chunks = [
        (args.code, probed, args.p, args.chunk, seed * 1_000_003 + i)
        for i in range(args.shots // args.chunk)
    ]
    with mp.Pool(args.workers) as pool:
        failures = [f for part in pool.map(collect, chunks) for f in part]
        total = len(chunks) * args.chunk
        print(
            f"{args.code} p={args.p}: {probed['name']}: {len(failures)} failures in {total} "
            f"shots (LER {len(failures) / total:.2e})"
        )
        for f in failures:
            lighter = f["output_weight"] < f["error_weight"]
            print(
                f"   {f['result']:6s} error weight {f['error_weight']:2d}, output weight "
                f"{f['output_weight']:2d}{'  (output lighter)' if lighter else ''}"
            )
        if failures:
            for spec in others:
                parts = [failures[k :: args.workers] for k in range(args.workers)]
                outcomes = pool.map(redecode, [(args.code, spec, args.p, part) for part in parts])
                merged = [None] * len(failures)
                for k, part in enumerate(outcomes):
                    for j, outcome in enumerate(part):
                        merged[k + j * args.workers] = outcome
                counts = {key: merged.count(key) for key in ("ok", "wrong", "unconv")}
                print(f"   redecoded by {spec['name']:<40} {counts}")
    if args.out:
        json.dump(
            dict(code=args.code, p=args.p, shots=total, decoder=probed["name"], failures=failures),
            open(args.out, "w"),
        )


if __name__ == "__main__":
    main()
