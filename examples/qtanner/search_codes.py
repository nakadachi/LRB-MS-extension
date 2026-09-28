"""Search random quantum Tanner code instances and print their parameters."""

import sys

import numpy as np
from qldpc import abstract

from qtanner_codes import build_qtanner, generates, group_table, random_tnc_subsets, satisfies_tnc
from distance import distance_upper_bound

# [6,3,3] local code: H = [P | I] with symmetric P, so C^perp is also [6,3,3].
P = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
H_A = np.hstack([P, np.eye(3, dtype=int)])
H_B = H_A[:, [0, 2, 4, 1, 3, 5]]

GROUPS = {
    "C16": lambda: abstract.CyclicGroup(16),
    "C20": lambda: abstract.CyclicGroup(20),
    "D8": lambda: abstract.DihedralGroup(4),
    "Q8": lambda: abstract.QuaternionGroup(),
    "A4": lambda: abstract.AlternatingGroup(4),
    "D12": lambda: abstract.DihedralGroup(6),
    "D16": lambda: abstract.DihedralGroup(8),
    "D20": lambda: abstract.DihedralGroup(10),
    "S4": lambda: abstract.SymmetricGroup(4),
    "D24": lambda: abstract.DihedralGroup(12),
    "SL(2,3)": lambda: abstract.SpecialLinearGroup(2, 3),
    "D30": lambda: abstract.DihedralGroup(15),
    "A5": lambda: abstract.AlternatingGroup(5),
}


def search(name, tries=40, seed=0, dist_trials=150):
    mul, inv, ident = group_table(GROUPS[name]())
    rng = np.random.default_rng(seed)
    found = []
    for t in range(tries):
        try:
            A, B = random_tnc_subsets(mul, inv, ident, 6, rng)
        except RuntimeError:
            return found
        if not (satisfies_tnc(mul, A, B) and generates(mul, A + B, ident)):
            continue
        code = build_qtanner(mul, A, B, H_A, H_B, name=name)
        if code.lx.shape[0] == 0:
            continue
        dx, _ = distance_upper_bound(code.hz, code.lz, trials=dist_trials, seed=t)
        dz, _ = distance_upper_bound(code.hx, code.lx, trials=dist_trials, seed=t)
        found.append((min(dx, dz), code.lx.shape[0], A, B, dx, dz))
        print(f"{name} |G|={len(mul)} try={t} n={code.n} k={code.lx.shape[0]} "
              f"dX<={dx} dZ<={dz} A={A} B={B}", flush=True)
    return found


if __name__ == "__main__":
    name, tries = sys.argv[1], int(sys.argv[2])
    search(name, tries=tries)
