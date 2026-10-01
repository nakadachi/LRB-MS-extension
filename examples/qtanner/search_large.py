"""Search larger quantum Tanner codes: PSL(2,q) groups and [6,3,3] or [8,4,4] local codes."""

import sys
import time

import numpy as np
from qldpc import abstract

from qtanner_codes import build_qtanner, generates, group_table, random_tnc_subsets, satisfies_tnc
from search_codes import H_A as H6_A, H_B as H6_B

# [8,4,4] extended Hamming code (self-dual); C_B is a column permutation of C_A
H8_A = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 0, 0, 1, 1],
                 [0, 1, 0, 1, 0, 1, 0, 1]])
H8_B = H8_A[:, [0, 3, 5, 6, 1, 2, 4, 7]]

LOCAL = {"633": (H6_A, H6_B, 6), "844": (H8_A, H8_B, 8)}
GROUPS = {
    "A5": lambda: abstract.AlternatingGroup(5),
    "PSL(2,7)": lambda: abstract.ProjectiveSpecialLinearGroup(2, 7),
    "PSL(2,8)": lambda: abstract.ProjectiveSpecialLinearGroup(2, 8),
}


def search(group, local, tries=6, seed=0):
    mul, inv, ident = group_table(GROUPS[group]())
    h_a, h_b, delta = LOCAL[local]
    rng = np.random.default_rng(seed)
    for t in range(tries):
        A, B = random_tnc_subsets(mul, inv, ident, delta, rng)
        if not (satisfies_tnc(mul, A, B) and generates(mul, A + B, ident)):
            continue
        t0 = time.time()
        code = build_qtanner(mul, A, B, h_a, h_b, name=f"{group}/{local}")
        print(f"{group} local={local} try={t} n={code.n} k={code.lx.shape[0]} "
              f"rows/vertex={code.rows_per_vertex_z} build={time.time() - t0:.1f}s A={A} B={B}", flush=True)


if __name__ == "__main__":
    search(sys.argv[1], sys.argv[2], int(sys.argv[3]))
