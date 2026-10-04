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

import mbp_ladder

WORKERS = 4
LEGS = [(0.75, 1.0), (1.0, 0.9), (0.9, 0.8), (0.6, 1.0), (1.0, 1.1)]
STRATEGIES = ("base", "ladder3", "lightest3", "lightest5")


def compare_chunk(args):
    """Per strategy: [failures, decoding seconds]."""
    name, p, shots, seed = args
    hx, hz, lx, lz, *_ = mbp_ladder.code(name)
    decoder = mbp_ladder.make(name, p, *LEGS[0])
    rng = np.random.default_rng(seed)
    stats = {strategy: [0, 0.0] for strategy in STRATEGIES}
    for _ in range(shots):
        error_x, error_z = mbp_ladder.sample(rng, hx.shape[1], p)
        syndrome_x, syndrome_z = hx @ error_z % 2, hz @ error_x % 2
        # per leg: None if unconverged, else (output weight, flips a logical)
        leg_results, leg_seconds = [], []
        for mu, alpha in LEGS:
            decoder.mu, decoder.alpha = mu, alpha
            start = time.perf_counter()
            decoded_x, decoded_z = decoder.decode(syndrome_x, syndrome_z)
            leg_seconds.append(time.perf_counter() - start)
            if not decoder.converge:
                leg_results.append(None)
                continue
            residual_x = (error_x + decoded_x) % 2
            residual_z = (error_z + decoded_z) % 2
            weight = int(((decoded_x | decoded_z) > 0).sum())
            wrong = bool(((lz @ residual_x) % 2).any() or ((lx @ residual_z) % 2).any())
            leg_results.append((weight, wrong))

        stats["base"][0] += leg_results[0] is None or leg_results[0][1]
        stats["base"][1] += leg_seconds[0]

        ladder_failed, ladder_seconds = True, 0.0
        for leg in range(3):
            ladder_seconds += leg_seconds[leg]
            if leg_results[leg] is not None:
                ladder_failed = leg_results[leg][1]
                break
        stats["ladder3"][0] += ladder_failed
        stats["ladder3"][1] += ladder_seconds

        for strategy, leg_count in (("lightest3", 3), ("lightest5", 5)):
            valid = [result for result in leg_results[:leg_count] if result is not None]
            # min over (weight, wrong): lightest output, a correct one preferred on ties
            stats[strategy][0] += min(valid)[1] if valid else True
            stats[strategy][1] += sum(leg_seconds[:leg_count])
    return stats


def main():
    name, p, shots = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    with mp.Pool(WORKERS) as pool:
        chunks = [(name, p, shots // WORKERS, 4242 + k) for k in range(WORKERS)]
        worker_stats = pool.map(compare_chunk, chunks)
    total_shots = (shots // WORKERS) * WORKERS
    print(f"{name} p={p}: {total_shots} shots, legs {LEGS}")
    for strategy in STRATEGIES:
        fails = sum(stats[strategy][0] for stats in worker_stats)
        seconds_per_shot = sum(stats[strategy][1] for stats in worker_stats) / total_shots
        print(
            f"  {strategy:<10} fails {fails:5d}  LER {fails / total_shots:.2e}  "
            f"{1e3 * seconds_per_shot:.2f} ms/shot",
            flush=True,
        )


if __name__ == "__main__":
    main()
