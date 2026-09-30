import multiprocessing as mp
from harness import *
from bb_codes import coset_groups
code="[[144,12,12]]"   # l=12, m=6
specs=[("greedy ell=6",dict(kind="lrbms",groups=6)),
 ("y-cosets (6)",dict(kind="lrbms",groups=("coset",1,1,1,6))),
 ("y^3-cosets (2)",dict(kind="lrbms",groups=("coset",1,1,3,2))),
 ("y^2-cosets (3)",dict(kind="lrbms",groups=("coset",1,1,2,3))),
 ("x^6-cosets (2)",dict(kind="lrbms",groups=("coset",6,2,1,1))),
 ("x^4-cosets (3)",dict(kind="lrbms",groups=("coset",4,3,1,1))),
 ("x^3-cosets (4)",dict(kind="lrbms",groups=("coset",3,4,1,1))),
 ("x^3,y^3 (8)",dict(kind="lrbms",groups=("coset",3,4,3,2))),
 ("x-cosets (12)",dict(kind="lrbms",groups=("coset",1,12,1,1))),
 ("x^6,y^2 (6)",dict(kind="lrbms",groups=("coset",6,2,2,3))),
 ("x^4,y^3 (6)",dict(kind="lrbms",groups=("coset",4,3,3,2))),
]
with mp.Pool(4) as pool:
    for p in [0.05]:
        for name,s in specs:
            r=simulate(pool,code,s,p,4000,10**9,250,11+int(p*1000))
            print(f"p={p} {name:>16}: LER={r['fails']/r['shots']:.3e} ({r['fails']}/{r['shots']}) unconv={r['unconverged']} {r['us_per_shot']:.0f}us",flush=True)
