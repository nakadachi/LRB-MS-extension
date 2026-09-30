"""Lower-bound the failure rate of a minimum-weight decoder by collecting light wrong-coset solutions."""
import multiprocessing as mp, sys
import numpy as np
from harness import get_code, make_decoder

def chunk(args):
    name, p, shots, seed = args
    code = get_code(name); hz = code.hz; lz = code.lz.astype(np.int64)
    decs = [make_decoder(dict(kind="bposd", osd_order=o), code, p) for o in (7, 40)]
    decs += [make_decoder(dict(kind="ensemble", ell=ell, count=4, stop="all", osd_method="osd_cs", osd_order=7), code, p)
             for ell in (8, 12)]
    rng = np.random.default_rng(seed)
    lb = tie = 0; fails = [0] * len(decs)
    for _ in range(shots):
        e = (rng.random(code.n) < p).astype(np.uint8); s = (hz @ e % 2).astype(np.uint8)
        lighter = eq = False
        for i, d in enumerate(decs):
            x = d.decode(s)
            if ((hz @ x) % 2 != s).any():
                fails[i] += 1; continue
            wrong = ((lz @ ((x + e) % 2)) % 2).any()
            fails[i] += bool(wrong)
            if wrong and x.sum() < e.sum(): lighter = True
            if wrong and x.sum() == e.sum(): eq = True
        lb += lighter; tie += (eq and not lighter)
    return lb, tie, fails, shots

if __name__ == "__main__":
    name, p, shots = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    with mp.Pool(4) as pool:
        res = pool.map(chunk, [(name, p, shots // 8, 99 + i) for i in range(8)])
    lb = sum(r[0] for r in res); tie = sum(r[1] for r in res); n = sum(r[3] for r in res)
    fails = np.sum([r[2] for r in res], axis=0)
    print(f"{name} p={p}: shots={n} MW lower bound={lb/n:.3e} (+ties/2 {(lb+tie/2)/n:.3e}) "
          f"BP-OSD7={fails[0]/n:.3e} BP-OSD40={fails[1]/n:.3e} ens8={fails[2]/n:.3e} ens12={fails[3]/n:.3e}", flush=True)
