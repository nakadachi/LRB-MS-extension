"""(mu, alpha)-relay ladders for the MBP4 + LRB-MS hybrid under depolarizing code-capacity noise.

A ladder is a list of (mu, alpha) legs. The first leg decodes every shot; each later leg is
fired only when all previous legs failed to converge. A shot fails if the final output does
not match the syndrome or flips a logical. Converged-but-wrong outputs end the ladder.

Modes
    fixed  : evaluate given ladders on the same shots
             python mbp_ladder.py fixed bb288 0.04 600000 "0.75,1.0;0.90,1.0;0.55,1.0"
    search : collect the base leg's failures, try a (mu, alpha) grid on them, build a ladder
             greedily on half of the failing shots and report it on the other half
             python mbp_ladder.py search bb288 0.04 600000 "0.75,1.0"
"""

import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from ldpc.lrbms_decoder import MbpLrbmsDecoder, overlap_check_groups

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "bb"))
sys.path.insert(0, str(HERE / "qtanner"))

# name: (builder, ell, max_iter, lrbms_order)
CODES = {
    "bb288": ("bb", "[[288,12,18]]", 6, 1000, 6),
    "bb288o1": ("bb", "[[288,12,18]]", 6, 1000, 1),   # LRB order 1, as in the screenshots
    "bb144": ("bb", "[[144,12,12]]", 6, 1000, 6),
    "qtC16": ("qt", "C16", 9, 40, 16),
    # stand-in for the screenshots' qt432_20: A4, [6,3,3] local codes, k = 20, d <= 16 (info-set bound)
    "qt432": ("qt432", None, 9, 40, 16),
}
_CACHE = {}


def code(name):
    if name in _CACHE:
        return _CACHE[name]
    kind, key, ell, it, order = CODES[name]
    if kind == "bb":
        from bb_codes import BBCode
        c = BBCode(key)
        xg, zg = overlap_check_groups(c.hx, ell), overlap_check_groups(c.hz, ell)
    elif kind == "qt432":
        from qtanner_codes import build_qtanner, group_table
        from search_432 import GROUPS12
        from search_codes import H_A
        mul, _, _ = group_table(GROUPS12["A4"]())
        c = build_qtanner(mul, [3, 4, 6, 8, 9, 11], [1, 2, 3, 7, 9, 10], H_A, H_A[:, [0, 5, 4, 1, 2, 3]])
        assert c.lx.shape[0] == 20
        xg, zg = c.vertex_groups("x"), c.vertex_groups("z")
    else:
        from benchmark import load_code
        c = load_code(dict(label="C16", group="C16", A=[1, 4, 5, 11, 12, 15], B=[2, 6, 7, 9, 10, 14]))
        xg, zg = c.vertex_groups("x"), c.vertex_groups("z")
    _CACHE[name] = (c.hx, c.hz, c.lx.astype(np.int64), c.lz.astype(np.int64), xg, zg, it, order)
    return _CACHE[name]


def sample(rng, n, p):
    r = rng.random(n)
    x = r < p / 3
    y = (r >= p / 3) & (r < 2 * p / 3)
    z = (r >= 2 * p / 3) & (r < p)
    return (x | y).astype(np.uint8), (z | y).astype(np.uint8)


def outcome(dec, hx, hz, lx, lz, ex, ez, sx, sz):
    dx, dz = dec.decode(sx, sz)
    if not dec.converge:
        return "unconv"
    rx, rz = (ex + dx) % 2, (ez + dz) % 2
    return "wrong" if ((lz @ rx) % 2).any() or ((lx @ rz) % 2).any() else "ok"


def make(name, p, mu, alpha):
    hx, hz, lx, lz, xg, zg, it, order = code(name)
    return MbpLrbmsDecoder(hx, hz, error_rate=p, x_groups=xg, z_groups=zg, max_iter=it, mu=mu,
                           alpha=alpha, lrbms_order=order, schedule="serial")


def run_ladders(args):
    """For each shot, run every ladder; record outcome and time of each."""
    name, p, shots, seed, ladders = args
    hx, hz, lx, lz, *_ = code(name)
    dec = make(name, p, *ladders[0][0])
    rng = np.random.default_rng(seed)
    res = [[0, 0, 0.0, [0] * len(lad)] for lad in ladders]   # fails, wrong, time, fires per leg
    for _ in range(shots):
        ex, ez = sample(rng, hx.shape[1], p)
        sx, sz = hx @ ez % 2, hz @ ex % 2
        for li, lad in enumerate(ladders):
            t0 = time.perf_counter()
            o = "unconv"
            for leg, (mu, alpha) in enumerate(lad):
                dec.mu, dec.alpha = mu, alpha
                res[li][3][leg] += 1
                o = outcome(dec, hx, hz, lx, lz, ex, ez, sx, sz)
                if o != "unconv":
                    break
            res[li][2] += time.perf_counter() - t0
            res[li][0] += o != "ok"
            res[li][1] += o == "wrong"
    return res


def base_failures(args):
    name, p, shots, seed, mu, alpha = args
    hx, hz, lx, lz, *_ = code(name)
    dec = make(name, p, mu, alpha)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(shots):
        ex, ez = sample(rng, hx.shape[1], p)
        sx, sz = hx @ ez % 2, hz @ ex % 2
        o = outcome(dec, hx, hz, lx, lz, ex, ez, sx, sz)
        if o != "ok":
            out.append((seed, i, o, ex, ez))
    return out


GRID_MU = (0.45, 0.55, 0.6, 0.65, 0.75, 0.9, 1.0)
GRID_ALPHA = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25)


def grid_outcomes(args):
    name, p, items = args
    hx, hz, lx, lz, *_ = code(name)
    dec = make(name, p, 0.75, 1.0)
    grid = [(m, a) for m in GRID_MU for a in GRID_ALPHA]
    rows = []
    for ex, ez in items:
        sx, sz = hx @ ez % 2, hz @ ex % 2
        row = []
        for mu, alpha in grid:
            dec.mu, dec.alpha = mu, alpha
            row.append(outcome(dec, hx, hz, lx, lz, ex, ez, sx, sz))
        rows.append(row)
    return rows


def parse_ladder(s):
    return [tuple(float(v) for v in leg.split(",")) for leg in s.split(";")]


def main():
    mode, name, p, shots = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
    w = 4
    if mode == "fixed":
        ladders = [parse_ladder(s) for s in sys.argv[5:]]
        with mp.Pool(w) as pool:
            parts = pool.map(run_ladders, [(name, p, shots // w, 4242 + k, ladders) for k in range(w)])
        n = (shots // w) * w
        print(f"{name} p={p}: {n} shots, I={code(name)[6]}")
        for li, lad in enumerate(ladders):
            fails = sum(r[li][0] for r in parts)
            wrong = sum(r[li][1] for r in parts)
            tm = sum(r[li][2] for r in parts) / n
            fires = [sum(r[li][3][k] for r in parts) for k in range(len(lad))]
            legs = " -> ".join(f"({m:g},{a:g})" for m, a in lad)
            print(f"  {legs:<40} fails {fails:5d} ({wrong} wrong)  LER {fails / n:.2e}  "
                  f"{1e3 * tm:.2f} ms/shot  legs fired {fires}", flush=True)
        return

    mu0, a0 = parse_ladder(sys.argv[5])[0]
    levels = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    with mp.Pool(w) as pool:
        fails = [f for ch in pool.map(base_failures, [(name, p, shots // w, 777 + k, mu0, a0) for k in range(w)])
                 for f in ch]
        unconv = [f for f in fails if f[2] == "unconv"]
        parts = [unconv[k::w] for k in range(w)]
        res = pool.map(grid_outcomes, [(name, p, [(f[3], f[4]) for f in part]) for part in parts])
    table = [None] * len(unconv)
    for k, rows in enumerate(res):
        for j, row in enumerate(rows):
            table[k + j * w] = row
    grid = [(m, a) for m in GRID_MU for a in GRID_ALPHA]
    n = (shots // w) * w
    train = [i for i in range(len(unconv)) if unconv[i][1] % 2 == 0]
    test = [i for i in range(len(unconv)) if unconv[i][1] % 2 == 1]
    wrong_test = sum(1 for f in fails if f[2] == "wrong" and f[1] % 2 == 1)
    print(f"{name} p={p}: {n} shots, base ({mu0:g},{a0:g}) failures {len(fails)} "
          f"({len(unconv)} unconverged, {len(fails) - len(unconv)} converged wrong)")
    solo = sorted(((sum(table[i][g] == "ok" for i in train), g) for g in range(len(grid))), reverse=True)
    print("  best single retries on training failures: " +
          ", ".join(f"({grid[g][0]:g},{grid[g][1]:g}) {c}/{len(train)}" for c, g in solo[:6]))
    ladder, remaining = [], list(train)
    for _ in range(levels):
        best = max(range(len(grid)), key=lambda g: sum(table[i][g] == "ok" for i in remaining))
        if not any(table[i][best] == "ok" for i in remaining):
            break
        ladder.append(best)
        remaining = [i for i in remaining if table[i][best] == "unconv"]
    half = n / 2
    unresolved, wrong = list(test), wrong_test
    print(f"  test half ({half:.0f} shots): leg 1 ({mu0:g},{a0:g}) LER {(len(unresolved) + wrong) / half:.2e} "
          f"({len(unresolved)} unconv + {wrong} wrong)")
    for lvl, g in enumerate(ladder, 2):
        nxt = []
        for i in unresolved:
            o = table[i][g]
            if o == "wrong":
                wrong += 1
            elif o == "unconv":
                nxt.append(i)
        unresolved = nxt
        print(f"    + leg {lvl} ({grid[g][0]:g},{grid[g][1]:g}): LER {(len(unresolved) + wrong) / half:.2e} "
              f"({len(unresolved)} unconv + {wrong} wrong)", flush=True)


if __name__ == "__main__":
    main()
