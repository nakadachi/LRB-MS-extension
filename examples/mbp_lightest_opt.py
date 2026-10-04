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

import mbp_ladder

WORKERS = 4
MUS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0)
ALPHAS = (0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5)
GRID = [(mu, alpha) for mu in MUS for alpha in ALPHAS]
NO_OUTPUT = 10**6  # weight recorded when a leg does not converge


def record_legs(args):
    """Per shot and leg: output weight, whether it flips a logical; per leg: total seconds."""
    name, p, shots, seed, legs = args
    hx, hz, lx, lz, *_ = mbp_ladder.code(name)
    decoder = mbp_ladder.make(name, p, *legs[0])
    rng = np.random.default_rng(seed)
    weights = np.full((shots, len(legs)), NO_OUTPUT, dtype=np.int64)
    wrong = np.zeros((shots, len(legs)), dtype=bool)
    seconds = np.zeros(len(legs))
    for shot in range(shots):
        error_x, error_z = mbp_ladder.sample(rng, hx.shape[1], p)
        syndrome_x, syndrome_z = hx @ error_z % 2, hz @ error_x % 2
        for leg, (mu, alpha) in enumerate(legs):
            decoder.mu, decoder.alpha = mu, alpha
            start = time.perf_counter()
            decoded_x, decoded_z = decoder.decode(syndrome_x, syndrome_z)
            seconds[leg] += time.perf_counter() - start
            if decoder.converge:
                weights[shot, leg] = int(((decoded_x | decoded_z) > 0).sum())
                residual_x = (error_x + decoded_x) % 2
                residual_z = (error_z + decoded_z) % 2
                flips = ((lz @ residual_x) % 2).any() or ((lx @ residual_z) % 2).any()
                wrong[shot, leg] = bool(flips)
    return weights, wrong, seconds


def lightest_failures(weights, wrong, selected):
    """Failures of 'lightest valid output among the selected legs' (ties: first selected)."""
    selected_weights = weights[:, selected]
    choice = np.argmin(selected_weights, axis=1)
    shots = np.arange(len(weights))
    valid = selected_weights[shots, choice] < NO_OUTPUT
    chosen_wrong = wrong[:, selected][shots, choice]
    return int((~valid | chosen_wrong).sum())


def gather(pool, name, p, shots, first_seed, legs):
    """Runs record_legs on WORKERS processes; returns (weights, wrong, seconds per shot)."""
    chunks = [(name, p, shots // WORKERS, first_seed + k, legs) for k in range(WORKERS)]
    parts = pool.map(record_legs, chunks)
    weights = np.vstack([part[0] for part in parts])
    wrong = np.vstack([part[1] for part in parts])
    seconds_per_shot = sum(part[2] for part in parts) / ((shots // WORKERS) * WORKERS)
    return weights, wrong, seconds_per_shot


def main():
    name, p = sys.argv[1], float(sys.argv[2])
    train_shots, test_shots, max_legs = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    with mp.Pool(WORKERS) as pool:
        weights, wrong, _ = gather(pool, name, p, train_shots, 100, GRID)
        selected = []
        for _ in range(max_legs):
            best = min(
                (g for g in range(len(GRID)) if g not in selected),
                key=lambda g: lightest_failures(weights, wrong, selected + [g]),
            )
            selected.append(best)
            print(
                f"train: legs {[GRID[g] for g in selected]} -> "
                f"{lightest_failures(weights, wrong, selected)} / {len(weights)}",
                flush=True,
            )
        legs = [GRID[g] for g in selected]
        base_leg = [(0.75, 1.0)]
        test_weights, test_wrong, test_seconds = gather(
            pool, name, p, test_shots, 9000, base_leg + legs
        )
    print(f"test ({len(test_weights)} fresh shots, p={p}):")
    base_fails = lightest_failures(test_weights, test_wrong, [0])
    print(f"  single leg (0.75,1): {base_fails} fails  {1e3 * test_seconds[0]:.2f} ms/shot")
    for leg_count in range(1, len(legs) + 1):
        selected = list(range(1, leg_count + 1))
        print(
            f"  lightest of {leg_count}: {legs[:leg_count]} -> "
            f"{lightest_failures(test_weights, test_wrong, selected)} fails  "
            f"{1e3 * test_seconds[1 : leg_count + 1].sum():.2f} ms/shot",
            flush=True,
        )


if __name__ == "__main__":
    main()
