"""Compare stopping rules for the Qulid grouping ensemble from one recorded pass.

For every shot, each of K groupings is decoded with Qulid alone (time, output, converged);
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

WORKERS = 4
MEMBER_COUNT, ELL, MSF, OSD_ORDER = 8, 8, 0.75, 7
MAX_ITER = int(sys.argv[4]) if len(sys.argv) > 4 else 100


def record_chunk(args):
    """Per shot: the true error weight and, per member, its Qulid and OSD results."""
    name, p, shots, seed = args
    code = get_code(name)
    hz, lz = code.hz, code.lz.astype(np.int64)
    groupings = permuted_overlap_groupings(hz, ELL, MEMBER_COUNT)
    settings = dict(error_rate=p, max_iter=MAX_ITER, ms_scaling_factor=MSF, schedule="serial")
    plain = [LrbmsDecoder(hz, check_groups=g, **settings) for g in groupings]
    with_osd = [
        LrbmsDecoder(hz, check_groups=g, osd_method="osd_cs", osd_order=OSD_ORDER, **settings)
        for g in groupings
    ]
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(shots):
        error = (rng.random(code.n) < p).astype(np.uint8)
        syndrome = (hz @ error % 2).astype(np.uint8)
        members = []
        for index in range(MEMBER_COUNT):
            start = time.perf_counter()
            output = plain[index].decode(syndrome).astype(np.uint8)
            lrb_seconds = time.perf_counter() - start
            converged = bool(plain[index].converge)
            if converged:
                wrong = bool(((lz @ ((output + error) % 2)) % 2).any())
            else:
                wrong = True
            member = dict(
                t_lrb=lrb_seconds,
                conv=converged,
                w=int(output.sum()),
                wrong=wrong,
                key=output.tobytes(),
            )
            if not converged:
                start = time.perf_counter()
                osd_output = with_osd[index].decode(syndrome).astype(np.uint8)
                # OSD time on top of the Qulid pass
                member["t_osd"] = max(time.perf_counter() - start - lrb_seconds, 0.0)
                member["w_osd"] = int(osd_output.sum())
                residual = (osd_output + error) % 2
                member["wrong_osd"] = bool(
                    ((hz @ residual) % 2).any() or ((lz @ residual) % 2).any()
                )
            members.append(member)
        records.append(dict(w_true=int(error.sum()), members=members))
    return records


def lightest_is_wrong(candidates):
    """Lowest-weight candidate (weight, wrong, ...) -> its wrong flag (first one on ties)."""
    return min(candidates, key=lambda c: c[0])[1]


def policy_all(record, member_count=MEMBER_COUNT, osd_all=True):
    seconds, candidates = 0.0, []
    for member in record["members"][:member_count]:
        seconds += member["t_lrb"]
        if member["conv"]:
            candidates.append((member["w"], member["wrong"]))
        elif osd_all:
            seconds += member["t_osd"]
            candidates.append((member["w_osd"], member["wrong_osd"]))
    if not candidates:  # osd_all=False and nothing converged: OSD on the last member
        member = record["members"][member_count - 1]
        seconds += member["t_osd"]
        candidates.append((member["w_osd"], member["wrong_osd"]))
    return lightest_is_wrong(candidates), seconds


def policy_stop(record, accept, member_count=MEMBER_COUNT):
    """Run members in order; stop when accept(converged candidates so far) is True.
    Returns the lowest-weight converged candidate, or OSD on the last member if none converged."""
    seconds, candidates = 0.0, []
    for member in record["members"][:member_count]:
        seconds += member["t_lrb"]
        if member["conv"]:
            candidates.append((member["w"], member["wrong"], member["key"]))
            if accept(candidates):
                break
    if not candidates:
        member = record["members"][member_count - 1]
        return member["wrong_osd"], seconds + member["t_osd"]
    return lightest_is_wrong(candidates), seconds


def policy_escalate(record, member_count=MEMBER_COUNT):
    """Accept member 0 if it converges; otherwise run members 1..k-1 with OSD on all and keep
    the lightest output."""
    first = record["members"][0]
    if first["conv"]:
        return first["wrong"], first["t_lrb"]
    seconds = first["t_lrb"] + first["t_osd"]
    candidates = [(first["w_osd"], first["wrong_osd"])]
    for member in record["members"][1:member_count]:
        seconds += member["t_lrb"]
        if member["conv"]:
            candidates.append((member["w"], member["wrong"]))
        else:
            seconds += member["t_osd"]
            candidates.append((member["w_osd"], member["wrong_osd"]))
    return lightest_is_wrong(candidates), seconds


def main():
    name, p, shots = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    with mp.Pool(WORKERS) as pool:
        chunks = [(name, p, shots // WORKERS, 77 + i) for i in range(WORKERS)]
        records = [record for chunk in pool.map(record_chunk, chunks) for record in chunk]
    n = get_code(name).n
    sigma = np.sqrt(n * p * (1 - p))

    def any_converged(candidates):
        return True

    def two_agree(candidates):
        return len({c[2] for c in candidates}) < len(candidates)  # some correction seen twice

    def weight_at_most(kappa):
        return lambda candidates: min(c[0] for c in candidates) <= n * p + kappa * sigma

    policies = {
        "all K=8, OSD on all (reference)": lambda r: policy_all(r),
        "all K=8, OSD only if none converged": lambda r: policy_all(r, osd_all=False),
        "all K=4, OSD on all": lambda r: policy_all(r, member_count=4),
        "first converged, OSD on last": lambda r: policy_stop(r, any_converged),
        "two members agree": lambda r: policy_stop(r, two_agree),
        "weight <= np": lambda r: policy_stop(r, weight_at_most(0.0)),
        "weight <= np + 1 sigma": lambda r: policy_stop(r, weight_at_most(1.0)),
        "weight <= np + 2 sigma": lambda r: policy_stop(r, weight_at_most(2.0)),
        "agree or weight <= np": lambda r: policy_stop(
            r, lambda c: two_agree(c) or weight_at_most(0.0)(c)
        ),
        "escalate: all K=8 only if first fails": lambda r: policy_escalate(r),
        "escalate: all K=4 only if first fails": lambda r: policy_escalate(r, member_count=4),
    }

    def single_seconds(record):
        first = record["members"][0]
        return first["t_lrb"] + (0 if first["conv"] else first["t_osd"])

    single = np.mean([single_seconds(record) for record in records])
    print(
        f"{name} p={p} max_iter={MAX_ITER} shots={len(records)}  "
        f"(single Qulid+OSD: {1e3 * single:.2f} ms/shot)"
    )
    for label, policy in policies.items():
        outcomes = [policy(record) for record in records]
        fails = sum(o[0] for o in outcomes)
        seconds = np.mean([o[1] for o in outcomes])
        print(
            f"  {label:<38} LER={fails / len(records):.2e} ({fails}/{len(records)})  "
            f"{1e3 * seconds:6.2f} ms/shot  ({seconds / single:4.1f}x single)",
            flush=True,
        )


if __name__ == "__main__":
    main()
