"""Search order-12 groups for [[432, 20]] quantum Tanner codes ([6,3,3] local codes, |A|=|B|=6).

Total no-conjugacy cannot hold for |G| = 12 with |A| = |B| = 6, so it is not required here.
"""

import sys

import numpy as np
from qldpc import abstract

from distance import distance_upper_bound
from qtanner_codes import build_qtanner, generates, group_table, random_symmetric_subset
from search_codes import H_A, H_B

GROUPS12 = {
    "C12": lambda: abstract.CyclicGroup(12),
    "C6xC2": lambda: abstract.AbelianGroup(6, 2),
    "D12": lambda: abstract.DihedralGroup(6),
    "A4": lambda: abstract.AlternatingGroup(4),
}


def main():
    tries = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    for name, make_group in GROUPS12.items():
        mul, inv, ident = group_table(make_group())
        rng = np.random.default_rng(1)
        for attempt in range(tries):
            A = random_symmetric_subset(mul, inv, ident, 6, rng)
            B = random_symmetric_subset(mul, inv, ident, 6, rng)
            if not generates(mul, A + B, ident):
                continue
            code = build_qtanner(mul, A, B, H_A, H_B, name=name)
            k = code.lx.shape[0]
            if k != 20:
                continue
            dx, _ = distance_upper_bound(code.hz, code.lz, trials=300, seed=attempt)
            dz, _ = distance_upper_bound(code.hx, code.lx, trials=300, seed=attempt)
            print(
                f"{name} try={attempt} n={code.n} k={k} dX<={dx} dZ<={dz} A={A} B={B}",
                flush=True,
            )


if __name__ == "__main__":
    main()
