"""LRB-MS with x-coset and overlapping x+y-coset groupings on a BB code.

Usage: python explore4.py "[[144,12,12]]" 0.04,0.05
"""

import multiprocessing as mp
import sys

from harness import get_code, simulate

SHOTS = 4000
OSD = dict(osd_method="osd_cs", osd_order=7)


def specs_for(code):
    x_cosets = ("coset", 1, code.l, 1, 1)
    y_cosets = ("coset", 1, 1, 1, code.m)
    both = ("union", x_cosets, y_cosets)
    return [
        ("BP+OSD-CS7", dict(kind="bposd")),
        ("x-cosets", dict(kind="lrbms", groups=x_cosets)),
        ("x+y cosets (overlap)", dict(kind="lrbms", groups=both)),
        ("x+y cosets +OSD", dict(kind="lrbms", groups=both, **OSD)),
        ("x+y cosets t=4 +OSD", dict(kind="lrbms", groups=both, t=4, **OSD)),
        ("x+y msf0.6 +OSD", dict(kind="lrbms", groups=both, msf=0.6, **OSD)),
        ("x+y msf0.5 +OSD", dict(kind="lrbms", groups=both, msf=0.5, **OSD)),
    ]


def main():
    code_name = sys.argv[1]
    ps = [float(value) for value in sys.argv[2].split(",")]
    specs = specs_for(get_code(code_name))
    with mp.Pool(4) as pool:
        for p in ps:
            for label, spec in specs:
                result = simulate(pool, code_name, spec, p, SHOTS, 10**9, 250, 11 + int(p * 1000))
                print(
                    f"{code_name} p={p} {label:>22}: "
                    f"LER={result['fails'] / result['shots']:.3e} "
                    f"({result['fails']}/{result['shots']}) "
                    f"unconv={result['unconverged']} {result['us_per_shot']:.0f}us",
                    flush=True,
                )


if __name__ == "__main__":
    main()
