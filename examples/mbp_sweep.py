"""Sweep (mu, alpha) for a single MBP4 + LRB-MS leg on identical shots (total depolarizing p).

Usage: python mbp_sweep.py bb144 0.06 20000 [ell]
"""
import multiprocessing as mp
import sys

import numpy as np

import mbp_ladder as M
from ldpc.lrbms_decoder import MbpLrbmsDecoder, overlap_check_groups

MUS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0)
ALPHAS = (0.7, 0.8, 0.9, 1.0, 1.1, 1.25)


def chunk(args):
    name, p, shots, seed, ell = args
    hx, hz, lx, lz, xg, zg, it, order = M.code(name)
    if ell:
        xg, zg = overlap_check_groups(hx, ell), overlap_check_groups(hz, ell)
    dec = MbpLrbmsDecoder(hx, hz, error_rate=p, x_groups=xg, z_groups=zg, max_iter=it, mu=0.75,
                          alpha=1.0, lrbms_order=order, schedule="serial")
    rng = np.random.default_rng(seed)
    grid = [(m, a) for m in MUS for a in ALPHAS]
    fails = np.zeros((len(grid), 2), dtype=int)   # [failures, wrong convergences]
    for _ in range(shots):
        ex, ez = M.sample(rng, hx.shape[1], p)
        sx, sz = hx @ ez % 2, hz @ ex % 2
        for g, (mu, a) in enumerate(grid):
            dec.mu, dec.alpha = mu, a
            o = M.outcome(dec, hx, hz, lx, lz, ex, ez, sx, sz)
            fails[g, 0] += o != "ok"
            fails[g, 1] += o == "wrong"
    return fails


if __name__ == "__main__":
    name, p, shots = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    ell = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    with mp.Pool(4) as pool:
        tot = sum(pool.map(chunk, [(name, p, shots // 4, 4242 + k, ell) for k in range(4)]))
    n = (shots // 4) * 4
    print(f"{name} p={p} ell={ell or 'default'}: {n} shots. failures (wrong convergences)")
    print("  mu \\ alpha " + "".join(f"{a:>12}" for a in ALPHAS))
    for i, mu in enumerate(MUS):
        row = tot[i * len(ALPHAS):(i + 1) * len(ALPHAS)]
        print(f"  {mu:<10}" + "".join(f"{f:>7} ({w:>3})" for f, w in row), flush=True)
