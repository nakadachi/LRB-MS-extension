"""Retry ladder for LRB-MS: on non-convergence, retry with another scaling, schedule or grouping.

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

WORKERS = 4

TANNER = {
    "C16": dict(label="C16", group="C16", A=[1, 4, 5, 11, 12, 15], B=[2, 6, 7, 9, 10, 14]),
    "A5/844": dict(
        label="A5/844",
        group="A5",
        local="844",
        A=[11, 14, 17, 26, 29, 31, 32, 53],
        B=[2, 3, 4, 30, 36, 44, 56, 58],
    ),
    "PSL27/633": dict(
        label="PSL27/633",
        group="PSL(2,7)",
        local="633",
        A=[81, 88, 121, 141, 161, 165],
        B=[9, 29, 64, 67, 113, 131],
    ),
}

BASE = dict(msf=0.75, schedule="serial", grouping=0)

# Prior given to the bits next to the checks the failed attempt left unsatisfied (their LLR is
# pushed towards 0): a classic trapping-set escape.
PERTURB_Q = (0.15, 0.25, 0.35, 0.45)
SCALING_FACTORS = (0.4, 0.5, 0.6, 0.7, 0.75, 0.85, 1.0)

_problem_cache = {}


def problem(name):
    """(H_Z, Z logicals, groupings): grouping 0 is the base one, the others are alternatives."""
    if name in _problem_cache:
        return _problem_cache[name]
    family, key = name.split(":", 1)
    if family == "tanner":
        from benchmark import load_code

        css = load_code(TANNER[key])
        groupings = [css.vertex_groups("z")]
    else:
        from harness import get_code

        css = get_code(key)
        groupings = permuted_overlap_groupings(css.hz, 8, 4, seed=0)
    _problem_cache[name] = (css.hz, css.lz.astype(np.int64), groupings)
    return _problem_cache[name]


def candidates(grouping_count):
    """All retry configurations: perturbed priors, then every (grouping, schedule, msf)."""
    configs = [dict(BASE, perturb=q) for q in PERTURB_Q]
    for grouping in range(grouping_count):
        for schedule in ("serial", "parallel"):
            for msf in SCALING_FACTORS:
                config = dict(msf=msf, schedule=schedule, grouping=grouping)
                if config != BASE:
                    configs.append(config)
    return configs


def make(config, hz, groupings, p):
    return LrbmsDecoder(
        hz,
        error_rate=p,
        check_groups=groupings[config["grouping"]],
        max_iter=100,
        ms_scaling_factor=config["msf"],
        lrbms_order=8,
        schedule=config["schedule"],
    )


def outcome(decoder, hz, lz, error, syndrome):
    """'ok', 'wrong' (converged to the wrong coset) or 'unconv'."""
    decoded = decoder.decode(syndrome)
    if not decoder.converge:
        return "unconv"
    return "wrong" if ((lz @ ((decoded + error) % 2)) % 2).any() else "ok"


def base_chunk(args):
    """Failing shots of the base decoder: (seed, shot, outcome, error)."""
    name, p, shots, seed = args
    hz, lz, groupings = problem(name)
    decoder = make(BASE, hz, groupings, p)
    rng = np.random.default_rng(seed)
    failures = []
    for shot in range(shots):
        error = (rng.random(hz.shape[1]) < p).astype(np.uint8)
        syndrome = (hz @ error % 2).astype(np.uint8)
        result = outcome(decoder, hz, lz, error, syndrome)
        if result != "ok":
            failures.append((seed, shot, result, error))
    return failures


def retry_chunk(args):
    """For each error, the outcome of every candidate retry configuration."""
    name, p, errors = args
    hz, lz, groupings = problem(name)
    configs = candidates(len(groupings))
    decoders = [make(config, hz, groupings, p) for config in configs]
    base_decoder = make(BASE, hz, groupings, p)
    table = []
    for error in errors:
        syndrome = (hz @ error % 2).astype(np.uint8)
        base_output = base_decoder.decode(syndrome)
        unsatisfied = np.flatnonzero((hz @ base_output + syndrome) % 2)
        if len(unsatisfied):
            suspect_bits = np.unique(hz[unsatisfied].indices)
        else:
            suspect_bits = np.array([], dtype=int)
        row = []
        for config, decoder in zip(configs, decoders):
            if "perturb" in config:
                channel = np.full(hz.shape[1], p)
                channel[suspect_bits] = config["perturb"]
                decoder.update_channel_probs(channel)
            row.append(outcome(decoder, hz, lz, error, syndrome))
        table.append(row)
    return table


def main():
    name, p, shots = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    levels = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    with mp.Pool(WORKERS) as pool:
        chunks = [(name, p, shots // WORKERS, 1000 + k) for k in range(WORKERS)]
        failures = [f for chunk in pool.map(base_chunk, chunks) for f in chunk]
        total_shots = (shots // WORKERS) * WORKERS
        unconverged = [f for f in failures if f[2] == "unconv"]
        base_wrong = len(failures) - len(unconverged)
        parts = [unconverged[k::WORKERS] for k in range(WORKERS)]
        part_tables = pool.map(retry_chunk, [(name, p, [f[3] for f in part]) for part in parts])
    # table[i][c]: outcome of candidate c on unconverged shot i (undo the round-robin split)
    table = [None] * len(unconverged)
    for worker, rows in enumerate(part_tables):
        for j, row in enumerate(rows):
            table[worker + j * WORKERS] = row
    _, _, groupings = problem(name)
    configs = candidates(len(groupings))
    train = [i for i in range(len(unconverged)) if unconverged[i][1] % 2 == 0]
    test = [i for i in range(len(unconverged)) if unconverged[i][1] % 2 == 1]
    # each half covers about total_shots / 2 shots; converged-wrong failures split the same way
    wrong_test = sum(1 for f in failures if f[2] == "wrong" and f[1] % 2 == 1)

    def rescues(c, shots_):
        return sum(table[i][c] == "ok" for i in shots_)

    solo = sorted(((rescues(c, train), c) for c in range(len(configs))), reverse=True)
    perturbed = [entry for entry in solo if "perturb" in configs[entry[1]]]
    print("  rescues of training failures by each candidate alone (top 6, plus perturbed):")
    for count, c in solo[:6] + perturbed:
        print(f"    {count:4d}/{len(train)}  {configs[c]}")
    ladder, remaining = [], list(train)
    for _ in range(levels):
        best = max(range(len(configs)), key=lambda c: rescues(c, remaining))
        if rescues(best, remaining) == 0:
            break
        ladder.append(best)
        remaining = [i for i in remaining if table[i][best] == "unconv"]

    print(
        f"{name} p={p}: {total_shots} shots, base failures {len(failures)} "
        f"({len(unconverged)} unconverged, {base_wrong} converged to the wrong coset)"
    )
    half = total_shots / 2
    unresolved, wrong = list(test), wrong_test
    print(f"  test half ({half:.0f} shots):")
    print(
        f"    level 0  base msf=0.75 serial grouping 0     : failure rate "
        f"{(len(unresolved) + wrong) / half:.2e}  ({len(unresolved)} unconv + {wrong} wrong)"
    )
    for level, c in enumerate(ladder, 1):
        config = configs[c]
        wrong += sum(table[i][c] == "wrong" for i in unresolved)
        unresolved = [i for i in unresolved if table[i][c] == "unconv"]
        if "perturb" in config:
            tag = f"perturb q={config['perturb']:<4}"
        else:
            tag = f"{config['schedule']:<8} grouping {config['grouping']}"
        print(
            f"    level {level}  msf={config['msf']:<4} {tag:<20}: "
            f"failure rate {(len(unresolved) + wrong) / half:.2e}  "
            f"({len(unresolved)} unconv + {wrong} wrong)",
            flush=True,
        )


if __name__ == "__main__":
    main()
