"""(mu, alpha)-relay ladders for the MBP4 + Qulid hybrid under depolarizing code-capacity noise.

A ladder is a list of (mu, alpha) legs. The first leg decodes every shot; each later leg is
fired only when all previous legs failed to converge. A shot fails if the final output does
not match the syndrome or flips a logical. Converged-but-wrong outputs end the ladder.

Modes
    fixed  : evaluate given ladders on the same shots
             python mbp_ladder.py fixed bb288 0.04 600000 "0.75,1.0;0.90,1.0;0.55,1.0"
    search : collect the base leg's failures, try a (mu, alpha) grid on them, build a ladder
             greedily on half of the failing shots and report it on the other half
             python mbp_ladder.py search bb288 0.04 600000 "0.75,1.0"

Noise convention for p (environment variable MBP_CONV):
    total    : total depolarizing p, p/3 per Pauli (default; all results in this repository)
    perpauli : p_X = p_Y = p_Z = p
    marginal : X and Z parts each flip with probability p (p/2 per Pauli)
    indep    : independent X and Z flips with probability p each
"""

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

from ldpc.lrbms_decoder import MbpLrbmsDecoder, overlap_check_groups

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "bb"))
sys.path.insert(0, str(HERE / "qtanner"))

WORKERS = 4
NOISE_CONVENTION = os.environ.get("MBP_CONV", "total")

# name: (family, key, ell, max_iter, lrbms_order)
CODES = {
    "bb288": ("bb", "[[288,12,18]]", 6, 1000, 6),
    "bb288o1": ("bb", "[[288,12,18]]", 6, 1000, 1),  # LRB order 1, as in the screenshots
    "bb144": ("bb", "[[144,12,12]]", 6, 1000, 6),
    "qtC16": ("qt", "C16", 9, 40, 16),
    # stand-in for the screenshots' qt432_20: A4, [6,3,3] local codes, k = 20, d <= 16
    "qt432": ("qt432", None, 9, 40, 16),
}

# (mu, alpha) grid tried on the base leg's failures in search mode
GRID_MU = (0.45, 0.55, 0.6, 0.65, 0.75, 0.9, 1.0)
GRID_ALPHA = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5)

_code_cache = {}


def code(name):
    """Returns (hx, hz, lx, lz, x_groups, z_groups, max_iter, lrbms_order) for a code name."""
    if name in _code_cache:
        return _code_cache[name]
    family, key, ell, max_iter, lrbms_order = CODES[name]
    if family == "bb":
        from bb_codes import BBCode

        css = BBCode(key)
        x_groups = overlap_check_groups(css.hx, ell)
        z_groups = overlap_check_groups(css.hz, ell)
    elif family == "qt432":
        from qtanner_codes import build_qtanner, group_table
        from search_432 import GROUPS12
        from search_codes import H_A

        multiplication, _, _ = group_table(GROUPS12["A4"]())
        css = build_qtanner(
            multiplication,
            [3, 4, 6, 8, 9, 11],
            [1, 2, 3, 7, 9, 10],
            H_A,
            H_A[:, [0, 5, 4, 1, 2, 3]],
        )
        assert css.lx.shape[0] == 20
        x_groups = css.vertex_groups("x")
        z_groups = css.vertex_groups("z")
    else:
        from benchmark import load_code

        css = load_code(
            dict(label="C16", group="C16", A=[1, 4, 5, 11, 12, 15], B=[2, 6, 7, 9, 10, 14])
        )
        x_groups = css.vertex_groups("x")
        z_groups = css.vertex_groups("z")
    _code_cache[name] = (
        css.hx,
        css.hz,
        css.lx.astype(np.int64),
        css.lz.astype(np.int64),
        x_groups,
        z_groups,
        max_iter,
        lrbms_order,
    )
    return _code_cache[name]


def pauli_probs(p):
    """(p_X, p_Y, p_Z) for physical error rate p under NOISE_CONVENTION."""
    if NOISE_CONVENTION == "total":
        return p / 3, p / 3, p / 3
    if NOISE_CONVENTION == "perpauli":
        return p, p, p
    if NOISE_CONVENTION == "marginal":
        return p / 2, p / 2, p / 2
    if NOISE_CONVENTION == "indep":
        return p * (1 - p), p * p, p * (1 - p)
    raise ValueError(NOISE_CONVENTION)


def sample(rng, qubit_count, p):
    """Samples a Pauli error; returns its (x-part, z-part)."""
    p_x, p_y, p_z = pauli_probs(p)
    draw = rng.random(qubit_count)
    is_x = draw < p_x
    is_y = (draw >= p_x) & (draw < p_x + p_y)
    is_z = (draw >= p_x + p_y) & (draw < p_x + p_y + p_z)
    return (is_x | is_y).astype(np.uint8), (is_z | is_y).astype(np.uint8)


def outcome(decoder, hx, hz, lx, lz, error_x, error_z, syndrome_x, syndrome_z):
    """'ok', 'wrong' (converged to the wrong coset) or 'unconv'."""
    decoded_x, decoded_z = decoder.decode(syndrome_x, syndrome_z)
    if not decoder.converge:
        return "unconv"
    residual_x = (error_x + decoded_x) % 2
    residual_z = (error_z + decoded_z) % 2
    flips_logical = ((lz @ residual_x) % 2).any() or ((lx @ residual_z) % 2).any()
    return "wrong" if flips_logical else "ok"


def make(name, p, mu, alpha):
    """An MbpLrbmsDecoder for the named code at error rate p."""
    hx, hz, lx, lz, x_groups, z_groups, max_iter, lrbms_order = code(name)
    return MbpLrbmsDecoder(
        hx,
        hz,
        channel=pauli_probs(p),
        x_groups=x_groups,
        z_groups=z_groups,
        max_iter=max_iter,
        mu=mu,
        alpha=alpha,
        lrbms_order=lrbms_order,
        schedule="serial",
    )


def run_ladders(args):
    """Runs every ladder on each shot; returns per ladder [fails, wrong, seconds, leg fires]."""
    name, p, shots, seed, ladders = args
    hx, hz, lx, lz, *_ = code(name)
    default_max_iter = code(name)[6]
    decoder = make(name, p, *ladders[0][0][:2])
    rng = np.random.default_rng(seed)
    stats = [[0, 0, 0.0, [0] * len(ladder)] for ladder in ladders]
    for _ in range(shots):
        error_x, error_z = sample(rng, hx.shape[1], p)
        syndrome_x, syndrome_z = hx @ error_z % 2, hz @ error_x % 2
        for ladder_index, ladder in enumerate(ladders):
            start = time.perf_counter()
            result = "unconv"
            for leg_index, leg in enumerate(ladder):
                # leg = (mu, alpha) or (mu, alpha, max_iter); default max_iter is the code's I
                decoder.mu, decoder.alpha = leg[0], leg[1]
                decoder.max_iter = int(leg[2]) if len(leg) > 2 else default_max_iter
                stats[ladder_index][3][leg_index] += 1
                result = outcome(decoder, hx, hz, lx, lz, error_x, error_z, syndrome_x, syndrome_z)
                if result != "unconv":
                    break
            stats[ladder_index][2] += time.perf_counter() - start
            stats[ladder_index][0] += result != "ok"
            stats[ladder_index][1] += result == "wrong"
    return stats


def base_failures(args):
    """Failing shots of a single (mu, alpha) decoder: (seed, shot, outcome, e_x, e_z)."""
    name, p, shots, seed, mu, alpha = args
    hx, hz, lx, lz, *_ = code(name)
    decoder = make(name, p, mu, alpha)
    rng = np.random.default_rng(seed)
    failures = []
    for shot in range(shots):
        error_x, error_z = sample(rng, hx.shape[1], p)
        syndrome_x, syndrome_z = hx @ error_z % 2, hz @ error_x % 2
        result = outcome(decoder, hx, hz, lx, lz, error_x, error_z, syndrome_x, syndrome_z)
        if result != "ok":
            failures.append((seed, shot, result, error_x, error_z))
    return failures


def grid_outcomes(args):
    """For each (e_x, e_z), the outcome of every (mu, alpha) on the grid."""
    name, p, errors = args
    hx, hz, lx, lz, *_ = code(name)
    decoder = make(name, p, 0.75, 1.0)
    grid = [(mu, alpha) for mu in GRID_MU for alpha in GRID_ALPHA]
    table = []
    for error_x, error_z in errors:
        syndrome_x, syndrome_z = hx @ error_z % 2, hz @ error_x % 2
        row = []
        for mu, alpha in grid:
            decoder.mu, decoder.alpha = mu, alpha
            row.append(outcome(decoder, hx, hz, lx, lz, error_x, error_z, syndrome_x, syndrome_z))
        table.append(row)
    return table


def parse_ladder(text):
    """'0.75,1.0;0.9,1.0,150' -> [(0.75, 1.0), (0.9, 1.0, 150.0)]."""
    return [tuple(float(value) for value in leg.split(",")) for leg in text.split(";")]


def format_leg(leg):
    return "(" + ",".join(f"{value:g}" for value in leg) + ")"


def run_fixed(name, p, shots, ladders):
    with mp.Pool(WORKERS) as pool:
        chunks = [(name, p, shots // WORKERS, 4242 + k, ladders) for k in range(WORKERS)]
        worker_stats = pool.map(run_ladders, chunks)
    total_shots = (shots // WORKERS) * WORKERS
    print(f"{name} p={p} conv={NOISE_CONVENTION}: {total_shots} shots, I={code(name)[6]}")
    for ladder_index, ladder in enumerate(ladders):
        stats = [worker[ladder_index] for worker in worker_stats]
        fails = sum(s[0] for s in stats)
        wrong = sum(s[1] for s in stats)
        seconds_per_shot = sum(s[2] for s in stats) / total_shots
        fires = [sum(s[3][leg] for s in stats) for leg in range(len(ladder))]
        legs = " -> ".join(format_leg(leg) for leg in ladder)
        print(
            f"  {legs:<40} fails {fails:5d} ({wrong} wrong)  LER {fails / total_shots:.2e}  "
            f"{1e3 * seconds_per_shot:.2f} ms/shot  legs fired {fires}",
            flush=True,
        )


def run_search(name, p, shots, mu0, alpha0, levels):
    with mp.Pool(WORKERS) as pool:
        chunks = [(name, p, shots // WORKERS, 777 + k, mu0, alpha0) for k in range(WORKERS)]
        failures = [f for chunk in pool.map(base_failures, chunks) for f in chunk]
        unconverged = [f for f in failures if f[2] == "unconv"]
        parts = [unconverged[k::WORKERS] for k in range(WORKERS)]
        part_tables = pool.map(
            grid_outcomes, [(name, p, [(f[3], f[4]) for f in part]) for part in parts]
        )
    # table[i][g]: outcome of grid point g on unconverged shot i (undo the round-robin split)
    table = [None] * len(unconverged)
    for worker, rows in enumerate(part_tables):
        for j, row in enumerate(rows):
            table[worker + j * WORKERS] = row
    grid = [(mu, alpha) for mu in GRID_MU for alpha in GRID_ALPHA]
    total_shots = (shots // WORKERS) * WORKERS
    # shots with even index train the ladder; odd ones test it
    train = [i for i in range(len(unconverged)) if unconverged[i][1] % 2 == 0]
    test = [i for i in range(len(unconverged)) if unconverged[i][1] % 2 == 1]
    wrong_test = sum(1 for f in failures if f[2] == "wrong" and f[1] % 2 == 1)
    print(
        f"{name} p={p}: {total_shots} shots, base ({mu0:g},{alpha0:g}) failures {len(failures)} "
        f"({len(unconverged)} unconverged, {len(failures) - len(unconverged)} converged wrong)"
    )

    def rescues(g, shots_):
        return sum(table[i][g] == "ok" for i in shots_)

    def new_wrong(g, shots_):
        return sum(table[i][g] == "wrong" for i in shots_)

    solo = sorted(((rescues(g, train), g) for g in range(len(grid))), reverse=True)
    print(
        "  best single retries on training failures: "
        + ", ".join(f"({grid[g][0]:g},{grid[g][1]:g}) {c}/{len(train)}" for c, g in solo[:6])
    )
    ladder, remaining = [], list(train)
    for _ in range(levels):
        # a wrong convergence ends the ladder, so score rescues minus new wrong convergences
        best = max(range(len(grid)), key=lambda g: rescues(g, remaining) - new_wrong(g, remaining))
        if rescues(best, remaining) == 0:
            break
        ladder.append(best)
        remaining = [i for i in remaining if table[i][best] == "unconv"]

    half = total_shots / 2
    unresolved, wrong = list(test), wrong_test
    print(
        f"  test half ({half:.0f} shots): leg 1 ({mu0:g},{alpha0:g}) "
        f"LER {(len(unresolved) + wrong) / half:.2e} ({len(unresolved)} unconv + {wrong} wrong)"
    )
    for level, g in enumerate(ladder, 2):
        wrong += sum(table[i][g] == "wrong" for i in unresolved)
        unresolved = [i for i in unresolved if table[i][g] == "unconv"]
        print(
            f"    + leg {level} ({grid[g][0]:g},{grid[g][1]:g}): "
            f"LER {(len(unresolved) + wrong) / half:.2e} "
            f"({len(unresolved)} unconv + {wrong} wrong)",
            flush=True,
        )


def main():
    mode, name, p, shots = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
    if mode == "fixed":
        run_fixed(name, p, shots, [parse_ladder(text) for text in sys.argv[5:]])
        return
    mu0, alpha0 = parse_ladder(sys.argv[5])[0][:2]
    levels = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    run_search(name, p, shots, mu0, alpha0, levels)


if __name__ == "__main__":
    main()
