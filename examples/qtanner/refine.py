"""Re-estimate the distance of one quantum Tanner code with more information-set trials.

Usage: python refine.py '{"group":"C16","A":[1,4,5,11,12,15],"B":[2,6,7,9,10,14]}' 2000
Prints the input entry extended with n, k, dX and dZ as one JSON line.
"""

import json
import sys

from distance import distance_upper_bound
from qtanner_codes import build_qtanner, group_table
from search_codes import GROUPS, H_A, H_B


def main():
    entry = json.loads(sys.argv[1])
    trials = int(sys.argv[2])
    mul, _, _ = group_table(GROUPS[entry["group"]]())
    code = build_qtanner(mul, entry["A"], entry["B"], H_A, H_B)
    dx, _ = distance_upper_bound(code.hz, code.lz, trials=trials, seed=11)
    dz, _ = distance_upper_bound(code.hx, code.lx, trials=trials, seed=12)
    result = dict(entry, n=code.n, k=int(code.lx.shape[0]), dX=dx, dZ=dz)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
