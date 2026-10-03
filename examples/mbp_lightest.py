"""Relay ladder vs 'lightest of several (mu, alpha) legs' for the MBP4 + LRB-MS hybrid.

The stopping ladder fires the next leg only when the previous one did not converge. The
'lightest of k' variant runs the first k legs on every shot and keeps the syndrome-valid output
with the fewest non-identity Paulis (the most likely error under depolarizing noise). It also
catches legs that converge to a heavier, wrong correction. Total depolarizing noise.

Usage: python mbp_lightest.py bb144 0.06 40000
"""

import multiprocessing as mp
import sys
import time

import numpy as np

import mbp_ladder as M

LEGS = [(0.75, 1.0), (1.0, 0.9), (0.9, 0.8), (0.6, 1.0), (1.0, 1.1)]


def chunk(args):
    name, p, shots, seed = args
    hx, hz, lx, lz, *_ = M.code(name)
    dec = M.make(name, p, *LEGS[0])
    rng = np.random.default_rng(seed)
    stats = {"base": [0, 0.0], "ladder3": [0, 0.0], "lightest3": [0, 0.0], "lightest5": [0, 0.0]}
    for _ in range(shots):
        ex, ez = M.sample(rng, hx.shape[1], p)
        sx, sz = hx @ ez % 2, hz @ ex % 2
        outs, times = [], []
        for mu, a in LEGS:
            dec.mu, dec.alpha = mu, a
            t0 = time.perf_counter()
            dx, dz = dec.decode(sx, sz)
            times.append(time.perf_counter() - t0)
            if dec.converge:
                rx, rz = (ex + dx) % 2, (ez + dz) % 2
                outs.append((int(((dx | dz) > 0).sum()), bool(((lz @ rx) % 2).any() or ((lx @ rz) % 2).any())))
            else:
                outs.append(None)
        stats["base"][0] += outs[0] is None or outs[0][1]
        stats["base"][1] += times[0]
        fail, t = True, 0.0
        for i in range(3):
            t += times[i]
            if outs[i] is not None:
                fail = outs[i][1]
                break
        stats["ladder3"][0] += fail
        stats["ladder3"][1] += t
        for key, k in (("lightest3", 3), ("lightest5", 5)):
            valid = [o for o in outs[:k] if o is not None]
            stats[key][0] += min(valid)[1] if valid else True
            stats[key][1] += sum(times[:k])
    return stats


if __name__ == "__main__":
    name, p, shots = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    w = 4
    with mp.Pool(w) as pool:
        res = pool.map(chunk, [(name, p, shots // w, 4242 + k) for k in range(w)])
    n = (shots // w) * w
    print(f"{name} p={p}: {n} shots, legs {LEGS}")
    for key in ("base", "ladder3", "lightest3", "lightest5"):
        f = sum(r[key][0] for r in res)
        t = sum(r[key][1] for r in res) / n
        print(f"  {key:<10} fails {f:5d}  LER {f / n:.2e}  {1e3 * t:.2f} ms/shot", flush=True)
