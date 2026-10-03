"""Choose the (mu, alpha) legs for 'lightest of k legs' decoding.

Records every grid setting's output (weight, wrong?) on training shots, picks legs greedily to
minimise failures of the lightest-valid-output rule, then evaluates the chosen legs on fresh
shots. Total depolarizing noise.

Usage: python mbp_lightest_opt.py bb144 0.06 20000 40000 5
"""
import multiprocessing as mp
import sys
import time

import numpy as np

import mbp_ladder as M

MUS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0)
ALPHAS = (0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5)
GRID = [(m, a) for m in MUS for a in ALPHAS]
BIG = 10 ** 6


def record(args):
    name, p, shots, seed, legs = args
    hx, hz, lx, lz, *_ = M.code(name)
    dec = M.make(name, p, *legs[0])
    rng = np.random.default_rng(seed)
    W = np.full((shots, len(legs)), BIG, dtype=np.int64)   # output weight (BIG = no valid output)
    WRONG = np.zeros((shots, len(legs)), dtype=bool)
    T = np.zeros(len(legs))
    for s in range(shots):
        ex, ez = M.sample(rng, hx.shape[1], p)
        sx, sz = hx @ ez % 2, hz @ ex % 2
        for g, (mu, a) in enumerate(legs):
            dec.mu, dec.alpha = mu, a
            t0 = time.perf_counter()
            dx, dz = dec.decode(sx, sz)
            T[g] += time.perf_counter() - t0
            if dec.converge:
                W[s, g] = int(((dx | dz) > 0).sum())
                rx, rz = (ex + dx) % 2, (ez + dz) % 2
                WRONG[s, g] = bool(((lz @ rx) % 2).any() or ((lx @ rz) % 2).any())
    return W, WRONG, T


def fails_of(W, WRONG, sel):
    """Failures of 'lightest valid output among sel' (ties: first in sel order)."""
    sub = W[:, sel]
    idx = np.argmin(sub, axis=1)
    rows = np.arange(len(W))
    valid = sub[rows, idx] < BIG
    wrong = WRONG[:, sel][rows, idx]
    return int((~valid | wrong).sum())


def gather(pool, name, p, shots, seed0, legs):
    parts = pool.map(record, [(name, p, shots // 4, seed0 + k, legs) for k in range(4)])
    return (np.vstack([q[0] for q in parts]), np.vstack([q[1] for q in parts]),
            sum(q[2] for q in parts) / ((shots // 4) * 4))


if __name__ == "__main__":
    name, p, n_train, n_test, kmax = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    with mp.Pool(4) as pool:
        W, WRONG, T = gather(pool, name, p, n_train, 100, GRID)
        sel = []
        for _ in range(kmax):
            best = min((g for g in range(len(GRID)) if g not in sel), key=lambda g: fails_of(W, WRONG, sel + [g]))
            sel.append(best)
            print(f"train: legs {[GRID[g] for g in sel]} -> {fails_of(W, WRONG, sel)} / {len(W)}", flush=True)
        legs = [GRID[g] for g in sel]
        base = [(0.75, 1.0)]
        W2, WRONG2, T2 = gather(pool, name, p, n_test, 9000, base + legs)
    n = len(W2)
    print(f"test ({n} fresh shots, p={p}):")
    print(f"  single leg (0.75,1): {fails_of(W2, WRONG2, [0])} fails  {1e3 * T2[0]:.2f} ms/shot")
    for k in range(1, len(legs) + 1):
        sel = list(range(1, k + 1))
        print(f"  lightest of {k}: {legs[:k]} -> {fails_of(W2, WRONG2, sel)} fails  "
              f"{1e3 * T2[1:k + 1].sum():.2f} ms/shot", flush=True)
