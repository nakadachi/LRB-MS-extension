"""Compare stopping rules for the LRB-MS grouping ensemble from one recorded pass.

For every shot, each of K groupings is decoded with LRB-MS alone (time, output, converged);
members that do not converge are also decoded with the OSD fallback (extra time, output).
Each policy is then replayed on the same records, so all policies see identical member
outputs and their costs are the summed member times they would have used.

Usage: python ensemble_policies.py "[[288,12,18]]" 0.06 3000 [max_iter]
"""

import multiprocessing as mp
import sys
import time

import numpy as np

from ldpc import LrbmsDecoder
from ldpc.lrbms_decoder import permuted_overlap_groupings

from harness import get_code

K, ELL, MSF, OSD_ORDER = 8, 8, 0.75, 7
MAX_ITER = int(sys.argv[4]) if len(sys.argv) > 4 else 100


def record_chunk(args):
    name, p, shots, seed = args
    code = get_code(name)
    hz, lz = code.hz, code.lz.astype(np.int64)
    groupings = permuted_overlap_groupings(hz, ELL, K)
    kw = dict(error_rate=p, max_iter=MAX_ITER, ms_scaling_factor=MSF, schedule="serial")
    plain = [LrbmsDecoder(hz, check_groups=g, **kw) for g in groupings]
    with_osd = [LrbmsDecoder(hz, check_groups=g, osd_method="osd_cs", osd_order=OSD_ORDER, **kw)
                for g in groupings]
    rng = np.random.default_rng(seed)
    recs = []
    for _ in range(shots):
        e = (rng.random(code.n) < p).astype(np.uint8)
        s = (hz @ e % 2).astype(np.uint8)
        members = []
        for i in range(K):
            t0 = time.perf_counter()
            d = plain[i].decode(s).astype(np.uint8)
            t_lrb = time.perf_counter() - t0
            conv = bool(plain[i].converge)
            m = dict(t_lrb=t_lrb, conv=conv, w=int(d.sum()),
                     wrong=bool(((lz @ ((d + e) % 2)) % 2).any()) if conv else True, key=d.tobytes())
            if not conv:
                t0 = time.perf_counter()
                o = with_osd[i].decode(s).astype(np.uint8)
                m["t_osd"] = max(time.perf_counter() - t0 - t_lrb, 0.0)  # OSD on top of the LRB pass
                m["w_osd"] = int(o.sum())
                m["wrong_osd"] = bool(((hz @ ((o + e) % 2)) % 2).any() or ((lz @ ((o + e) % 2)) % 2).any())
            members.append(m)
        recs.append(dict(w_true=int(e.sum()), members=members))
    return recs


def pick(cands):
    """Lowest-weight candidate (weight, wrong) -> wrong flag."""
    return min(cands, key=lambda c: c[0])[1]


def policy_all(r, k=K, osd_all=True):
    t, cands = 0.0, []
    for m in r["members"][:k]:
        t += m["t_lrb"]
        if m["conv"]:
            cands.append((m["w"], m["wrong"]))
        elif osd_all:
            t += m["t_osd"]
            cands.append((m["w_osd"], m["wrong_osd"]))
    if not cands:  # osd_all=False and nothing converged: OSD on the last member
        m = r["members"][k - 1]
        t += m["t_osd"]
        cands.append((m["w_osd"], m["wrong_osd"]))
    return pick(cands), t


def policy_stop(r, accept, k=K):
    """Run members in order; stop when accept(converged candidates so far) is True.
    Returns the lowest-weight converged candidate, or OSD on the last member if none converged."""
    t, cands = 0.0, []
    for m in r["members"][:k]:
        t += m["t_lrb"]
        if m["conv"]:
            cands.append((m["w"], m["wrong"], m["key"]))
            if accept(cands):
                break
    if not cands:
        m = r["members"][k - 1]
        return m["wrong_osd"], t + m["t_osd"]
    return min(cands, key=lambda c: c[0])[1], t


def policy_escalate(r, k=K):
    """Accept member 0 if it converges; otherwise run members 1..k-1 with OSD on all, keep the lightest."""
    m0 = r["members"][0]
    if m0["conv"]:
        return m0["wrong"], m0["t_lrb"]
    t, cands = m0["t_lrb"] + m0["t_osd"], [(m0["w_osd"], m0["wrong_osd"])]
    for m in r["members"][1:k]:
        t += m["t_lrb"]
        if m["conv"]:
            cands.append((m["w"], m["wrong"]))
        else:
            t += m["t_osd"]
            cands.append((m["w_osd"], m["wrong_osd"]))
    return pick(cands), t


def main():
    name, p, shots = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    with mp.Pool(4) as pool:
        recs = [r for chunk in pool.map(record_chunk, [(name, p, shots // 4, 77 + i) for i in range(4)])
                for r in chunk]
    n = get_code(name).n
    sigma = np.sqrt(n * p * (1 - p))

    def first(c):
        return True

    def agree2(c):
        return len({x[2] for x in c}) < len(c)  # some correction seen twice

    def weight_ok(kappa):
        return lambda c: min(x[0] for x in c) <= n * p + kappa * sigma

    policies = {
        "all K=8, OSD on all (reference)": lambda r: policy_all(r),
        "all K=8, OSD only if none converged": lambda r: policy_all(r, osd_all=False),
        "all K=4, OSD on all": lambda r: policy_all(r, k=4),
        "first converged, OSD on last": lambda r: policy_stop(r, first),
        "two members agree": lambda r: policy_stop(r, agree2),
        "weight <= np": lambda r: policy_stop(r, weight_ok(0.0)),
        "weight <= np + 1 sigma": lambda r: policy_stop(r, weight_ok(1.0)),
        "weight <= np + 2 sigma": lambda r: policy_stop(r, weight_ok(2.0)),
        "agree or weight <= np": lambda r: policy_stop(r, lambda c: agree2(c) or weight_ok(0.0)(c)),
        "escalate: all K=8 only if first fails": lambda r: policy_escalate(r),
        "escalate: all K=4 only if first fails": lambda r: policy_escalate(r, k=4),
    }
    single = np.mean([r["members"][0]["t_lrb"] + (0 if r["members"][0]["conv"] else r["members"][0]["t_osd"])
                      for r in recs])
    print(f"{name} p={p} max_iter={MAX_ITER} shots={len(recs)}  (single LRB-MS+OSD: {1e3 * single:.2f} ms/shot)")
    for label, f in policies.items():
        out = [f(r) for r in recs]
        fails = sum(o[0] for o in out)
        tm = np.mean([o[1] for o in out])
        print(f"  {label:<38} LER={fails / len(recs):.2e} ({fails}/{len(recs)})  "
              f"{1e3 * tm:6.2f} ms/shot  ({tm / single:4.1f}x single)", flush=True)


if __name__ == "__main__":
    main()
