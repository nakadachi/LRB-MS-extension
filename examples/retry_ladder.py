"""Retry ladder for LRB-MS: on non-convergence, retry with a different scaling / schedule / grouping.

1. Decode N shots with the base decoder; keep the failing shots.
2. Split the failing shots into a training and a test half (by shot index parity).
3. On the training failures, run every candidate retry configuration. Build the ladder greedily:
   each level is the candidate that rescues (converges to the correct coset) the most training
   shots still unconverged after the previous levels.
4. Replay the ladder on the test failures. Report the failure rate after each level, counting
   both unconverged shots and converged-but-wrong shots (a retry only fires on non-convergence).

Usage:
    python retry_ladder.py tanner:C16 0.06 40000
    python retry_ladder.py bb:[[288,12,18]] 0.05 40000
"""

import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

from ldpc import LrbmsDecoder
from ldpc.lrbms_decoder import permuted_overlap_groupings

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "qtanner"))
sys.path.insert(0, str(HERE / "bb"))

TANNER = {
    "C16": dict(label="C16", group="C16", A=[1, 4, 5, 11, 12, 15], B=[2, 6, 7, 9, 10, 14]),
    "A5/844": dict(label="A5/844", group="A5", local="844", A=[11, 14, 17, 26, 29, 31, 32, 53],
                   B=[2, 3, 4, 30, 36, 44, 56, 58]),
    "PSL27/633": dict(label="PSL27/633", group="PSL(2,7)", local="633",
                      A=[81, 88, 121, 141, 161, 165], B=[9, 29, 64, 67, 113, 131]),
}
_CACHE = {}


def problem(name):
    """(H_Z, Z logicals, list of groupings: index 0 = base grouping, others = alternatives)."""
    if name in _CACHE:
        return _CACHE[name]
    kind, key = name.split(":", 1)
    if kind == "tanner":
        from benchmark import load_code
        c = load_code(TANNER[key])
        groupings = [c.vertex_groups("z")]
    else:
        from harness import get_code
        c = get_code(key)
        groupings = permuted_overlap_groupings(c.hz, 8, 4, seed=0)
    _CACHE[name] = (c.hz, c.lz.astype(np.int64), groupings)
    return _CACHE[name]


BASE = dict(msf=0.75, schedule="serial", grouping=0)


PERTURB_Q = (0.15, 0.25, 0.35, 0.45)


def candidates(num_groupings):
    # perturbed retries: base settings, but bits next to the checks the failed attempt left
    # unsatisfied get prior q (their LLR is pushed towards 0) -- classic trapping-set escape
    out = [dict(BASE, perturb=q) for q in PERTURB_Q]
    for g in range(num_groupings):
        for sched in ("serial", "parallel"):
            for msf in (0.4, 0.5, 0.6, 0.7, 0.75, 0.85, 1.0):
                c = dict(msf=msf, schedule=sched, grouping=g)
                if c != BASE:
                    out.append(c)
    return out


def make(cfg, hz, groupings, p):
    return LrbmsDecoder(hz, error_rate=p, check_groups=groupings[cfg["grouping"]], max_iter=100,
                        ms_scaling_factor=cfg["msf"], lrbms_order=8, schedule=cfg["schedule"])


def outcome(dec, hz, lz, e, s):
    """'ok', 'wrong' (converged to the wrong coset) or 'unconv'."""
    d = dec.decode(s)
    if not dec.converge:
        return "unconv"
    return "wrong" if ((lz @ ((d + e) % 2)) % 2).any() else "ok"


def base_chunk(args):
    name, p, shots, seed = args
    hz, lz, groupings = problem(name)
    dec = make(BASE, hz, groupings, p)
    rng = np.random.default_rng(seed)
    fails = []
    for i in range(shots):
        e = (rng.random(hz.shape[1]) < p).astype(np.uint8)
        s = (hz @ e % 2).astype(np.uint8)
        o = outcome(dec, hz, lz, e, s)
        if o != "ok":
            fails.append((seed, i, o, e))
    return fails


def retry_chunk(args):
    name, p, items = args
    hz, lz, groupings = problem(name)
    cands = candidates(len(groupings))
    decs = [make(c, hz, groupings, p) for c in cands]
    base = make(BASE, hz, groupings, p)
    rows = []
    for e in items:
        s = (hz @ e % 2).astype(np.uint8)
        d0 = base.decode(s)
        unsat = np.flatnonzero((hz @ d0 + s) % 2)
        bits = np.unique(hz[unsat].indices) if len(unsat) else np.array([], dtype=int)
        row = []
        for c, d in zip(cands, decs):
            if "perturb" in c:
                ch = np.full(hz.shape[1], p)
                ch[bits] = c["perturb"]
                d.update_channel_probs(ch)
            row.append(outcome(d, hz, lz, e, s))
        rows.append(row)
    return rows


def main():
    name, p, shots = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    levels = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    w = 4
    with mp.Pool(w) as pool:
        fails = [f for ch in pool.map(base_chunk, [(name, p, shots // w, 1000 + k) for k in range(w)])
                 for f in ch]
        n_shots = (shots // w) * w
        unconv = [f for f in fails if f[2] == "unconv"]
        wrong0 = len(fails) - len(unconv)
        parts = [unconv[k::w] for k in range(w)]
        res = pool.map(retry_chunk, [(name, p, [f[3] for f in part]) for part in parts])
    table = [None] * len(unconv)
    for k, part_rows in enumerate(res):
        for j, row in enumerate(part_rows):
            table[k + j * w] = row
    hz, _, groupings = problem(name)
    cands = candidates(len(groupings))
    train = [i for i in range(len(unconv)) if unconv[i][1] % 2 == 0]
    test = [i for i in range(len(unconv)) if unconv[i][1] % 2 == 1]
    # each half covers about n_shots/2 shots; converged-wrong base failures split the same way
    wrong_test = sum(1 for f in fails if f[2] == "wrong" and f[1] % 2 == 1)

    solo = sorted(((sum(table[i][c] == "ok" for i in train), c) for c in range(len(cands))), reverse=True)
    print("  rescues of training failures by each candidate alone (top 6, plus perturbed):")
    for cnt, c in solo[:6] + [x for x in solo if "perturb" in cands[x[1]]]:
        print(f"    {cnt:4d}/{len(train)}  {cands[c]}")
    ladder, remaining = [], list(train)
    for _ in range(levels):
        best = max(range(len(cands)), key=lambda c: sum(table[i][c] == "ok" for i in remaining))
        if not any(table[i][best] == "ok" for i in remaining):
            break
        ladder.append(best)
        remaining = [i for i in remaining if table[i][best] == "unconv"]

    print(f"{name} p={p}: {n_shots} shots, base failures {len(fails)} "
          f"({len(unconv)} unconverged, {wrong0} converged to the wrong coset)")
    half = n_shots / 2
    unresolved, wrong = list(test), wrong_test
    print(f"  test half ({half:.0f} shots):")
    print(f"    level 0  base msf=0.75 serial grouping 0     : failure rate "
          f"{(len(unresolved) + wrong) / half:.2e}  ({len(unresolved)} unconv + {wrong} wrong)")
    for lvl, c in enumerate(ladder, 1):
        cfg = cands[c]
        nxt = []
        for i in unresolved:
            o = table[i][c]
            if o == "wrong":
                wrong += 1
            elif o == "unconv":
                nxt.append(i)
        unresolved = nxt
        tag = f"perturb q={cfg['perturb']:<4}" if "perturb" in cfg else f"{cfg['schedule']:<8} grouping {cfg['grouping']}"
        print(f"    level {lvl}  msf={cfg['msf']:<4} {tag:<20}: "
              f"failure rate {(len(unresolved) + wrong) / half:.2e}  "
              f"({len(unresolved)} unconv + {wrong} wrong)", flush=True)


if __name__ == "__main__":
    main()
