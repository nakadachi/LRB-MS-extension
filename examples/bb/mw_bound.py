"""Lower-bound the failure rate of a minimum-weight decoder by collecting light wrong-coset
solutions found by BP+OSD and Qulid ensembles.

Usage: python mw_bound.py "[[144,12,12]]" 0.05 40000
"""

import multiprocessing as mp
import sys

import numpy as np

from harness import get_code, make_decoder

WORKERS = 4
CHUNKS = 8


def bound_chunk(args):
    """Returns (lighter, ties, failures per decoder, shots).

    A shot counts as 'lighter' if some decoder found a wrong-coset correction strictly lighter
    than the true error (a minimum-weight decoder must fail on it), and as a 'tie' if the
    lightest such correction has the same weight as the error.
    """
    name, p, shots, seed = args
    code = get_code(name)
    hz, lz = code.hz, code.lz.astype(np.int64)
    decoders = [make_decoder(dict(kind="bposd", osd_order=order), code, p) for order in (7, 40)]
    decoders += [
        make_decoder(
            dict(kind="ensemble", ell=ell, count=4, stop="all", osd_method="osd_cs", osd_order=7),
            code,
            p,
        )
        for ell in (8, 12)
    ]
    rng = np.random.default_rng(seed)
    lighter_count = tie_count = 0
    fails = [0] * len(decoders)
    for _ in range(shots):
        error = (rng.random(code.n) < p).astype(np.uint8)
        syndrome = (hz @ error % 2).astype(np.uint8)
        found_lighter = found_tie = False
        for index, decoder in enumerate(decoders):
            correction = decoder.decode(syndrome)
            if ((hz @ correction) % 2 != syndrome).any():
                fails[index] += 1
                continue
            wrong = ((lz @ ((correction + error) % 2)) % 2).any()
            fails[index] += bool(wrong)
            if wrong and correction.sum() < error.sum():
                found_lighter = True
            if wrong and correction.sum() == error.sum():
                found_tie = True
        lighter_count += found_lighter
        tie_count += found_tie and not found_lighter
    return lighter_count, tie_count, fails, shots


def main():
    name, p, shots = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    with mp.Pool(WORKERS) as pool:
        chunks = [(name, p, shots // CHUNKS, 99 + i) for i in range(CHUNKS)]
        results = pool.map(bound_chunk, chunks)
    lighter = sum(r[0] for r in results)
    ties = sum(r[1] for r in results)
    total_shots = sum(r[3] for r in results)
    fails = np.sum([r[2] for r in results], axis=0)
    print(
        f"{name} p={p}: shots={total_shots} MW lower bound={lighter / total_shots:.3e} "
        f"(+ties/2 {(lighter + ties / 2) / total_shots:.3e}) "
        f"BP-OSD7={fails[0] / total_shots:.3e} BP-OSD40={fails[1] / total_shots:.3e} "
        f"ens8={fails[2] / total_shots:.3e} ens12={fails[3] / total_shots:.3e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
