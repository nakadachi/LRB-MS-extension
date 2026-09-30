import multiprocessing as mp, sys
from harness import *
code=sys.argv[1]; ps=[float(x) for x in sys.argv[2].split(",")]; shots=int(sys.argv[3])
l,m=get_code(code).l,get_code(code).m
X=("coset",1,l,1,1); Y=("coset",1,1,1,m)
osd=dict(osd_method="osd_cs",osd_order=7)
specs=[("BP (min-sum)",dict(kind="bp")),
 ("BP+OSD-CS7",dict(kind="bposd")),
 ("LRB-MS greedy8",dict(kind="lrbms",groups=8)),
 ("LRB-MS x-cosets",dict(kind="lrbms",groups=X)),
 ("LRB-MS greedy8 +OSD",dict(kind="lrbms",groups=8,**osd)),
 ("ens2 greedy8 all +OSD",dict(kind="ensemble",ell=8,count=2,stop="all",**osd)),
 ("ens4 greedy8 first +OSD",dict(kind="ensemble",ell=8,count=4,stop="first",**osd)),
 ("ens4 greedy8 all +OSD",dict(kind="ensemble",ell=8,count=4,stop="all",**osd)),
 ("ens8 greedy8 all +OSD",dict(kind="ensemble",ell=8,count=8,stop="all",**osd)),
 ("ens x,y-cosets+3 greedy8 +OSD",dict(kind="ensemble",ell=8,count=3,extra=[X,Y],stop="all",**osd)),
 ("ens4 greedy12 all +OSD",dict(kind="ensemble",ell=12,count=4,stop="all",**osd)),
]
with mp.Pool(4) as pool:
    for p in ps:
        for name,s in specs:
            r=simulate(pool,code,s,p,shots,10**9,250,11+int(p*1000))
            print(f"{code} p={p} {name:>30}: LER={r['fails']/r['shots']:.3e} ({r['fails']}/{r['shots']}) unconv={r['unconverged']} {r['us_per_shot']:.0f}us",flush=True)
