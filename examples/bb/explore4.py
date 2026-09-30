import multiprocessing as mp, sys
from harness import *
code=sys.argv[1]; ps=[float(x) for x in sys.argv[2].split(",")]; l,m=get_code(code).l,get_code(code).m
X=("coset",1,l,1,1); Y=("coset",1,1,1,m)
osd=dict(osd_method="osd_cs",osd_order=7)
specs=[("BP+OSD-CS7",dict(kind="bposd")),
 ("x-cosets",dict(kind="lrbms",groups=X)),
 ("x+y cosets (overlap)",dict(kind="lrbms",groups=("union",X,Y))),
 ("x+y cosets +OSD",dict(kind="lrbms",groups=("union",X,Y),**osd)),
 ("x+y cosets t=4 +OSD",dict(kind="lrbms",groups=("union",X,Y),t=4,**osd)),
 ("x+y msf0.6 +OSD",dict(kind="lrbms",groups=("union",X,Y),msf=0.6,**osd)),
 ("x+y msf0.5 +OSD",dict(kind="lrbms",groups=("union",X,Y),msf=0.5,**osd)),
]
with mp.Pool(4) as pool:
    for p in ps:
        for name,s in specs:
            r=simulate(pool,code,s,p,4000,10**9,250,11+int(p*1000))
            print(f"{code} p={p} {name:>22}: LER={r['fails']/r['shots']:.3e} ({r['fails']}/{r['shots']}) unconv={r['unconverged']} {r['us_per_shot']:.0f}us",flush=True)
