import sys, json
from search_codes import *
entry = json.loads(sys.argv[1]); trials = int(sys.argv[2])
mul, inv, ident = group_table(GROUPS[entry["group"]]())
code = build_qtanner(mul, entry["A"], entry["B"], H_A, H_B)
dx, _ = distance_upper_bound(code.hz, code.lz, trials=trials, seed=11)
dz, _ = distance_upper_bound(code.hx, code.lx, trials=trials, seed=12)
print(json.dumps(dict(entry, n=code.n, k=int(code.lx.shape[0]), dX=dx, dZ=dz)), flush=True)
