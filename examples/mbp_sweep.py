"""Sweep (mu, alpha) for a single MBP4 + Qulid leg on identical shots (total depolarizing p).

Usage: python mbp_sweep.py bb144 0.06 20000 [ell]
"""

import multiprocessing as mp
import sys

import numpy as np

from ldpc.lrbms_decoder import MbpLrbmsDecoder, overlap_check_groups

import mbp_ladder

WORKERS = 4
MUS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0)
ALPHAS = (0.7, 0.8, 0.9, 1.0, 1.1, 1.25)


def sweep_chunk(args):
    """Per grid point (mu-major order): [failures, wrong convergences]."""
    name, p, shots, seed, ell = args
    hx, hz, lx, lz, x_groups, z_groups, max_iter, lrbms_order = mbp_ladder.code(name)
    if ell:
        x_groups = overlap_check_groups(hx, ell)
        z_groups = overlap_check_groups(hz, ell)
    decoder = MbpLrbmsDecoder(
        hx,
        hz,
        error_rate=p,
        x_groups=x_groups,
        z_groups=z_groups,
        max_iter=max_iter,
        mu=0.75,
        alpha=1.0,
        lrbms_order=lrbms_order,
        schedule="serial",
    )
    rng = np.random.default_rng(seed)
    grid = [(mu, alpha) for mu in MUS for alpha in ALPHAS]
    counts = np.zeros((len(grid), 2), dtype=int)
    for _ in range(shots):
        error_x, error_z = mbp_ladder.sample(rng, hx.shape[1], p)
        syndrome_x, syndrome_z = hx @ error_z % 2, hz @ error_x % 2
        for g, (mu, alpha) in enumerate(grid):
            decoder.mu, decoder.alpha = mu, alpha
            result = mbp_ladder.outcome(
                decoder, hx, hz, lx, lz, error_x, error_z, syndrome_x, syndrome_z
            )
            counts[g, 0] += result != "ok"
            counts[g, 1] += result == "wrong"
    return counts


def main():
    name, p, shots = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    ell = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    with mp.Pool(WORKERS) as pool:
        chunks = [(name, p, shots // WORKERS, 4242 + k, ell) for k in range(WORKERS)]
        counts = sum(pool.map(sweep_chunk, chunks))
    total_shots = (shots // WORKERS) * WORKERS
    print(
        f"{name} p={p} ell={ell or 'default'}: {total_shots} shots. failures (wrong convergences)"
    )
    print("  mu \\ alpha " + "".join(f"{alpha:>12}" for alpha in ALPHAS))
    for i, mu in enumerate(MUS):
        row = counts[i * len(ALPHAS) : (i + 1) * len(ALPHAS)]
        cells = "".join(f"{fails:>7} ({wrong:>3})" for fails, wrong in row)
        print(f"  {mu:<10}" + cells, flush=True)


if __name__ == "__main__":
    main()
