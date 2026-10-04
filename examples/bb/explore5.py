"""BP, BP+OSD, single LRB-MS and LRB-MS grouping ensembles on a BB code.

Usage: python explore5.py "[[144,12,12]]" 0.04,0.05 4000
"""

import multiprocessing as mp
import sys

from harness import get_code, simulate

OSD = dict(osd_method="osd_cs", osd_order=7)


def specs_for(code):
    x_cosets = ("coset", 1, code.l, 1, 1)
    y_cosets = ("coset", 1, 1, 1, code.m)
    return [
        ("BP (min-sum)", dict(kind="bp")),
        ("BP+OSD-CS7", dict(kind="bposd")),
        ("LRB-MS greedy8", dict(kind="lrbms", groups=8)),
        ("LRB-MS x-cosets", dict(kind="lrbms", groups=x_cosets)),
        ("LRB-MS greedy8 +OSD", dict(kind="lrbms", groups=8, **OSD)),
        ("ens2 greedy8 all +OSD", dict(kind="ensemble", ell=8, count=2, stop="all", **OSD)),
        ("ens4 greedy8 first +OSD", dict(kind="ensemble", ell=8, count=4, stop="first", **OSD)),
        ("ens4 greedy8 all +OSD", dict(kind="ensemble", ell=8, count=4, stop="all", **OSD)),
        ("ens8 greedy8 all +OSD", dict(kind="ensemble", ell=8, count=8, stop="all", **OSD)),
        (
            "ens x,y-cosets+3 greedy8 +OSD",
            dict(kind="ensemble", ell=8, count=3, extra=[x_cosets, y_cosets], stop="all", **OSD),
        ),
        ("ens4 greedy12 all +OSD", dict(kind="ensemble", ell=12, count=4, stop="all", **OSD)),
    ]


def main():
    code_name = sys.argv[1]
    ps = [float(value) for value in sys.argv[2].split(",")]
    shots = int(sys.argv[3])
    specs = specs_for(get_code(code_name))
    with mp.Pool(4) as pool:
        for p in ps:
            for label, spec in specs:
                result = simulate(pool, code_name, spec, p, shots, 10**9, 250, 11 + int(p * 1000))
                print(
                    f"{code_name} p={p} {label:>30}: "
                    f"LER={result['fails'] / result['shots']:.3e} "
                    f"({result['fails']}/{result['shots']}) "
                    f"unconv={result['unconverged']} {result['us_per_shot']:.0f}us",
                    flush=True,
                )


if __name__ == "__main__":
    main()
