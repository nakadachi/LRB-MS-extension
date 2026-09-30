# LRB-MS: Least-Reliable-Basis Min-Sum decoding with generalized checks

`LrbmsDecoder` groups the rows of a parity-check matrix into *generalized checks*
(GCs) of `ell` rows, and runs min-sum message passing between variable nodes and GCs.
Each GC returns max-log extrinsic LLRs over its local syndrome coset. These are approximated by a
candidate list built from its least-reliable basis (LRB-MS-`t`), or computed exactly with
a 2^`ell`-state syndrome trellis (`gc_method="trellis"`, a reference baseline).
With one row per GC it reduces exactly to (normalised) min-sum BP.

The C++ core lives in `src_cpp/lrbms.hpp` and the Python bindings in `ldpc.lrbms_decoder`.

## Installation

Python 3.10 or newer and a C++ compiler are required.

```bash
git clone https://github.com/nakadachi/LRB-MS-extension.git
cd LRB-MS-extension
pip install -e .
```

The package installs under the import name `ldpc`.

## Quickstart

```python
import numpy as np
from ldpc import LrbmsDecoder
from ldpc.lrbms_decoder import overlap_check_groups, consecutive_check_groups

decoder = LrbmsDecoder(
    H,                       # np.ndarray or scipy.sparse parity-check matrix
    error_rate=0.05,         # or error_channel=[...]
    check_groups=9,          # int ell (greedy overlap grouping) or an explicit list of row lists
    lrbms_order=8,           # LRB-MS-t: order-2 re-encodings among the t least-reliable MRB positions
    gc_method="lrbms",       # or "trellis" (exact min-sum per GC, ell <= 22)
    schedule="serial",       # "parallel" (flooding) or "serial" (layered over GCs)
    max_iter=100,
    ms_scaling_factor=0.75,
    osd_method="off",        # optional fallback when LRB-MS does not converge: "osd_0", "osd_cs", "osd_e"
    osd_order=0,
)
correction = decoder.decode(syndrome)
decoder.converge, decoder.iterations, decoder.log_prob_ratios, decoder.osd_used
```

### Choosing the generalized checks

- `check_groups=None` gives one row per GC, which is plain min-sum BP.
- `check_groups=ell` (an integer) uses `overlap_check_groups(H, ell)`. Each group is grown
  greedily from rows that share the most columns.
- `consecutive_check_groups(m, ell)` makes blocks of consecutive rows. Use it when neighbouring
  rows already form a component code (GLDPC or Tanner-type constructions).
- You can also pass an explicit list of row lists. Every nonzero row must be covered.
- Groups should partition the rows. Putting a row in two groups counts its evidence twice and
  makes decoding much worse.

### Ensemble over groupings

`LrbmsEnsembleDecoder` runs LRB-MS once per check grouping and returns the syndrome-valid output
with the lowest channel cost. Different groupings put the GC boundaries in different places, so
their failures are only weakly correlated. This helps most on codes without a natural local-code
structure, such as bivariate bicycle codes (see the benchmark below).

```python
from ldpc.lrbms_decoder import LrbmsEnsembleDecoder

decoder = LrbmsEnsembleDecoder(
    H, error_rate=0.05,
    ell=8, num_groupings=8,  # greedy groupings grown from permuted row orders
                             # (or pass groupings=[...] explicitly)
    stop="first",            # "first": stop at the first converged grouping; "all": run all, keep the cheapest
    osd_members="last",      # OSD fallback only on the last grouping ("all": on every grouping)
    osd_method="osd_cs", osd_order=7,
    max_iter=100, ms_scaling_factor=0.75, schedule="serial",
)
correction = decoder.decode(syndrome)
decoder.converge, decoder.member
```

### Sinter

For stim/sinter workflows use `ldpc.sinter_decoders.SinterLrbmsDecoder`.

### Tests

```bash
pytest python_test/test_lrbms.py
```

The tests check four things:
- `ell = 1` reproduces min-sum BP exactly.
- The trellis matches brute-force max-log.
- LRB-MS is exact whenever its candidate list covers the coset.
- The OSD fallback works.
- The ensemble returns the cheapest valid member output.

## Benchmark: quantum Tanner codes

We compared LRB-MS with min-sum BP and BP+OSD on four Leverrier–Zémor quantum Tanner codes.
Each one is built on a group `G` with generating sets `|A| = |B| = 6` and `[6,3,3]` local codes.
Every vertex of the Cayley complex contributes a 9-row local tensor code, and LRB-MS uses one GC
per vertex. `overlap_check_groups(H, 9)` recovers exactly these groups.

The noise is code-capacity bit flips: independent X errors with probability `p`, decoded from
`H_Z`. A shot fails if the residual has a nonzero syndrome or is a nontrivial logical. At each
`(code, p)`, every decoder decodes the same error samples. A point stops at 100 failures or
20 000 shots (3 000 shots for the trellis). All decoders use a serial schedule, at most 100
iterations, and min-sum scaling 0.75, the value a sweep found best for both BP and LRB-MS.
Distances are upper bounds from a random information-set search. Times are per shot on one core.

![Quantum Tanner code benchmark](examples/qtanner/qtanner_benchmark.png)

### Logical error rate per shot

`0 (/N)` means no failures in `N` shots.

**[[576,32,≤16]] (C16)**

| decoder | 0.02 | 0.03 | 0.04 | 0.05 | 0.06 | 0.07 | 0.08 | 0.09 | ms/shot @ p=0.05 |
|---|---|---|---|---|---|---|---|---|---|
| BP (min-sum) | 4.5e-04 | 2.4e-03 | 8.7e-03 | 4.3e-02 | 1.4e-01 | 3.8e-01 | 6.1e-01 | 8.5e-01 | 0.84 |
| BP+OSD-CS7 | 4.0e-04 | 1.9e-03 | 7.7e-03 | 3.7e-02 | 1.2e-01 | 3.5e-01 | 5.8e-01 | 8.3e-01 | 1.25 |
| BP(PS)+OSD-CS7 | 5.0e-04 | 2.6e-03 | 9.3e-03 | 3.6e-02 | 1.1e-01 | 3.1e-01 | 5.6e-01 | 8.3e-01 | 4.07 |
| **LRB-MS-0** | 0 (/20000) | 0 (/20000) | 2.5e-04 | 2.0e-03 | 1.6e-02 | 9.2e-02 | 3.0e-01 | 6.2e-01 | 0.50 |
| **LRB-MS-8** | 0 (/20000) | 5.0e-05 | 1.0e-04 | 1.4e-03 | 1.2e-02 | 8.4e-02 | 2.8e-01 | 5.8e-01 | 0.60 |
| **LRB-MS-8+OSD-CS7** | 0 (/20000) | 5.0e-05 | 1.0e-04 | 1.1e-03 | 1.1e-02 | 8.7e-02 | 2.7e-01 | 5.8e-01 | 0.68 |
| Exact GC (trellis) | 0 (/3200) | 0 (/3200) | 0 (/3200) | 9.4e-04 | 1.3e-02 | 9.5e-02 | 3.3e-01 | 6.3e-01 | 6.25 |

**[[864,22,≤18]] (SL(2,3))**

| decoder | 0.02 | 0.03 | 0.04 | 0.05 | 0.06 | 0.07 | 0.08 | 0.09 | ms/shot @ p=0.05 |
|---|---|---|---|---|---|---|---|---|---|
| BP (min-sum) | 0 (/20000) | 3.5e-04 | 1.4e-03 | 1.2e-02 | 6.6e-02 | 3.4e-01 | 6.8e-01 | 8.9e-01 | 1.08 |
| BP+OSD-CS7 | 0 (/20000) | 1.0e-04 | 9.5e-04 | 8.4e-03 | 5.7e-02 | 2.9e-01 | 6.3e-01 | 8.7e-01 | 1.35 |
| BP(PS)+OSD-CS7 | 3.0e-04 | 1.8e-03 | 4.2e-03 | 1.3e-02 | 6.4e-02 | 3.4e-01 | 6.0e-01 | 8.4e-01 | 5.32 |
| **LRB-MS-0** | 0 (/20000) | 0 (/20000) | 0 (/20000) | 1.0e-04 | 3.4e-03 | 5.2e-02 | 2.7e-01 | 6.2e-01 | 0.76 |
| **LRB-MS-8** | 0 (/20000) | 0 (/20000) | 0 (/20000) | 5.0e-05 | 2.9e-03 | 4.9e-02 | 2.5e-01 | 6.1e-01 | 0.86 |
| **LRB-MS-8+OSD-CS7** | 0 (/20000) | 0 (/20000) | 0 (/20000) | 5.0e-05 | 2.5e-03 | 4.5e-02 | 2.3e-01 | 6.1e-01 | 0.90 |
| Exact GC (trellis) | 0 (/3200) | 0 (/3200) | 0 (/3200) | 0 (/3200) | 3.4e-03 | 7.2e-02 | 3.0e-01 | 6.5e-01 | 9.17 |

**[[1080,24,≤18]] (D30)**

| decoder | 0.02 | 0.03 | 0.04 | 0.05 | 0.06 | 0.07 | 0.08 | 0.09 | ms/shot @ p=0.05 |
|---|---|---|---|---|---|---|---|---|---|
| BP (min-sum) | 0 (/20000) | 4.0e-04 | 2.7e-03 | 1.3e-02 | 5.7e-02 | 2.8e-01 | 6.2e-01 | 9.5e-01 | 1.35 |
| BP+OSD-CS7 | 0 (/20000) | 2.0e-04 | 1.8e-03 | 8.9e-03 | 4.1e-02 | 2.3e-01 | 5.9e-01 | 9.3e-01 | 1.53 |
| BP(PS)+OSD-CS7 | 4.5e-04 | 2.2e-03 | 7.8e-03 | 2.0e-02 | 5.1e-02 | 2.7e-01 | 6.0e-01 | 9.0e-01 | 7.63 |
| **LRB-MS-0** | 0 (/20000) | 0 (/20000) | 0 (/20000) | 1.0e-04 | 2.2e-03 | 3.7e-02 | 2.5e-01 | 6.6e-01 | 0.87 |
| **LRB-MS-8** | 0 (/20000) | 0 (/20000) | 0 (/20000) | 5.0e-05 | 1.4e-03 | 2.8e-02 | 2.2e-01 | 6.3e-01 | 1.06 |
| **LRB-MS-8+OSD-CS7** | 0 (/20000) | 0 (/20000) | 0 (/20000) | 0 (/20000) | 1.0e-03 | 2.4e-02 | 2.1e-01 | 6.1e-01 | 1.09 |
| Exact GC (trellis) | 0 (/3200) | 0 (/3200) | 0 (/3200) | 0 (/3200) | 2.5e-03 | 3.9e-02 | 2.6e-01 | 6.7e-01 | 11.45 |

**[[2160,24]] (A5)**

| decoder | 0.02 | 0.03 | 0.04 | 0.05 | 0.06 | 0.07 | 0.08 | 0.09 | ms/shot @ p=0.05 |
|---|---|---|---|---|---|---|---|---|---|
| BP (min-sum) | 0 (/20000) | 1.0e-04 | 9.5e-04 | 5.1e-03 | 2.5e-02 | 1.8e-01 | 6.9e-01 | 9.7e-01 | 2.45 |
| BP+OSD-CS7 | 0 (/20000) | 0 (/20000) | 2.0e-04 | 9.0e-04 | 9.7e-03 | 1.3e-01 | 6.5e-01 | 9.6e-01 | 4.32 |
| BP(PS)+OSD-CS7 | 1.4e-03 | 5.6e-03 | 1.8e-02 | 3.9e-02 | 8.6e-02 | 2.6e-01 | 6.6e-01 | 9.5e-01 | 79.48 |
| **LRB-MS-0** | 0 (/20000) | 0 (/20000) | 0 (/20000) | 0 (/20000) | 5.0e-05 | 4.0e-03 | 1.3e-01 | 6.5e-01 | 1.88 |
| **LRB-MS-8** | 0 (/20000) | 0 (/20000) | 0 (/20000) | 0 (/20000) | 5.0e-05 | 2.7e-03 | 1.1e-01 | 6.1e-01 | 2.21 |
| **LRB-MS-8+OSD-CS7** | 0 (/20000) | 0 (/20000) | 0 (/20000) | 0 (/20000) | 0 (/20000) | 2.7e-03 | 1.1e-01 | 5.9e-01 | 2.27 |
| Exact GC (trellis) | 0 (/3200) | 0 (/3200) | 0 (/3200) | 0 (/3200) | 0 (/3200) | 4.7e-03 | 1.7e-01 | 7.1e-01 | 26.40 |


### Findings

- **LRB-MS lowers the logical error rate by 20–200×** compared with BP and BP+OSD-CS7, wherever
  both can be measured. The rise in failures moves from p ≈ 0.03–0.04 to p ≈ 0.06–0.07.
- **It is as fast as or faster than the baselines.** For p ≤ 0.06, LRB-MS-8 costs about the same
  as plain min-sum BP, and it is cheaper than BP+OSD from p = 0.05 upwards.
- **OSD does not rescue BP on these codes.** When BP fails to converge, OSD-CS7 returns a
  correction that satisfies the syndrome but is much heavier than the true error, so the result
  is a logical error. Product-sum BP gives OSD worse soft information and is 4–40× slower.
- **The grouping is where the gain comes from.** LRB-MS-0 (order-1 re-encodings only) already
  captures most of it. Order 8 improves the error rate by a further 1.2–1.5× for about 20% more time.
- **The exact per-GC trellis matches LRB-MS-8 but runs 5–30× slower**, so the LRB candidate list
  is not the bottleneck. An OSD fallback after LRB-MS helps by at most 1.2×.

The code construction, benchmark scripts and raw data (`results.json`) are in
[`examples/qtanner`](examples/qtanner/README.md).

## Benchmark: bivariate bicycle codes

We ran the same code-capacity bit-flip benchmark on the bivariate bicycle (BB) codes of
Bravyi et al. (arXiv:2308.07915): [[72,12,6]], [[144,12,12]] and [[288,12,18]].
Decoding uses `H_Z`, and every decoder sees the same error samples.
A point stops at 100 failures or 20 000 shots.
All decoders use a serial schedule, at most 100 iterations and min-sum scaling 0.75.
The ensembles use 8 greedy `ell = 8` groupings.

![BB code benchmark](examples/bb/bb_benchmark.png)

### Logical error rate per shot

`0 (/N)` means no failures in `N` shots.

**[[72,12,6]]**

| decoder | 0.03 | 0.04 | 0.05 | 0.06 | 0.07 | 0.08 | ms/shot @ p=0.05 |
|---|---|---|---|---|---|---|---|
| BP (min-sum) | 3.0e-02 | 8.3e-02 | 1.5e-01 | 2.4e-01 | 3.7e-01 | 4.7e-01 | 0.06 |
| BP+OSD-CS7 | 3.0e-02 | 7.9e-02 | 1.5e-01 | 2.4e-01 | 3.6e-01 | 4.5e-01 | 0.07 |
| BP+OSD-CS40 | 3.0e-02 | 7.9e-02 | 1.5e-01 | 2.4e-01 | 3.6e-01 | 4.5e-01 | 0.09 |
| LRB-MS ell=8 + OSD-CS7 | 3.3e-02 | 8.0e-02 | 1.7e-01 | 2.4e-01 | 3.8e-01 | 4.6e-01 | 0.09 |
| **Ensemble ×8, first** | 3.5e-02 | 8.0e-02 | 1.7e-01 | 2.4e-01 | 3.8e-01 | 4.6e-01 | 0.11 |
| **Ensemble ×8, all** | 3.0e-02 | 7.2e-02 | 1.6e-01 | 2.3e-01 | 3.6e-01 | 4.6e-01 | 0.49 |

**[[144,12,12]]**

| decoder | 0.03 | 0.04 | 0.05 | 0.06 | 0.07 | 0.08 | ms/shot @ p=0.05 |
|---|---|---|---|---|---|---|---|
| BP (min-sum) | 3.3e-03 | 2.0e-02 | 5.8e-02 | 1.7e-01 | 2.8e-01 | 4.5e-01 | 0.10 |
| BP+OSD-CS7 | 1.4e-03 | 8.3e-03 | 2.5e-02 | 1.0e-01 | 1.8e-01 | 3.1e-01 | 0.12 |
| BP+OSD-CS40 | 1.3e-03 | 7.8e-03 | 2.2e-02 | 9.0e-02 | 1.6e-01 | 2.9e-01 | 0.16 |
| LRB-MS ell=8 + OSD-CS7 | 1.5e-03 | 1.1e-02 | 2.9e-02 | 1.1e-01 | 2.0e-01 | 2.9e-01 | 0.22 |
| **Ensemble ×8, first** | 1.3e-03 | 9.0e-03 | 2.6e-02 | 1.0e-01 | 1.8e-01 | 2.8e-01 | 0.39 |
| **Ensemble ×8, all** | 1.0e-03 | 7.8e-03 | 2.3e-02 | 8.6e-02 | 1.7e-01 | 2.8e-01 | 1.71 |

**[[288,12,18]]**

| decoder | 0.03 | 0.04 | 0.05 | 0.06 | 0.07 | 0.08 | ms/shot @ p=0.05 |
|---|---|---|---|---|---|---|---|
| BP (min-sum) | 2.0e-04 | 3.7e-03 | 2.4e-02 | 8.8e-02 | 2.4e-01 | 4.9e-01 | 0.17 |
| BP+OSD-CS7 | 0 (/20000) | 4.0e-04 | 4.6e-03 | 2.7e-02 | 7.6e-02 | 2.3e-01 | 0.20 |
| BP+OSD-CS40 | 0 (/20000) | 3.5e-04 | 3.2e-03 | 2.0e-02 | 6.2e-02 | 2.1e-01 | 0.26 |
| LRB-MS ell=8 + OSD-CS7 | 0 (/20000) | 9.0e-04 | 6.8e-03 | 3.2e-02 | 1.0e-01 | 2.8e-01 | 0.39 |
| **Ensemble ×8, first** | 0 (/20000) | 5.0e-04 | 2.4e-03 | 2.1e-02 | 6.8e-02 | 2.3e-01 | 0.64 |
| **Ensemble ×8, all** | 0 (/20000) | 2.5e-04 | 1.6e-03 | 1.5e-02 | 5.2e-02 | 1.8e-01 | 3.05 |


### How much can be gained?

To size the headroom, we estimated how an optimal minimum-weight decoder would do on the same
samples. A shot counts as a sure failure if any decoder found a valid correction in the wrong
logical class that is lighter than the true error. Wrong-class corrections of equal weight count
as half a failure, since no minimum-weight rule can break the tie.

| code, p | min-weight estimate | BP+OSD-CS7 | BP+OSD-CS40 | ensemble ×4, all |
|---|---|---|---|---|
| [[144,12,12]], 0.03 | ≈ 8e-4 | 8.8e-4 | 7.5e-4 | 1.1e-3 |
| [[144,12,12]], 0.05 | ≈ 1.9e-2 | 3.0e-2 | 2.8e-2 | 2.6e-2 |
| [[288,12,18]], 0.05 | ≈ 4e-4 | 4.6e-3 | 3.8e-3 | 1.7e-3 |
| [[288,12,18]], 0.07 | ≈ 2.4e-2 | 9.2e-2 | 8.3e-2 | 7.5e-2 |

### Findings

- **On BB codes a single LRB-MS decoder does not beat BP+OSD.** Unlike the Tanner codes, BB
  checks do not combine into strong local codes. Groupings cut the error rate of plain min-sum BP
  by up to about 2×, but stay well above BP+OSD. This holds for greedy groupings and for groupings
  that follow the code's algebra (cosets of the x and y shifts). The exact per-GC trellis is no
  better either. With an OSD fallback, one LRB-MS decoder is
  slightly worse than BP+OSD-CS7.
- **The failures are fixable by regrouping.** When LRB-MS fails, it either does not converge or
  converges to a valid but heavier correction in the wrong logical class. A different grouping
  often succeeds on the same syndrome.
- **The grouping ensemble is the improvement.** On [[288,12,18]] at p = 0.05, "ensemble ×8, all"
  lowers the logical error rate to 1.6e-3, compared with 4.6e-3 for BP+OSD-CS7 and 3.2e-3 for
  BP+OSD-CS40. That is 1.9× better than the stronger baseline. It costs about 12× the
  decode time of BP+OSD-CS40.
- **"Ensemble ×8, first" is the cheaper option.** It stops at the first converged grouping and runs
  OSD only on the last. It beats BP+OSD-CS40 by about 1.3× at p = 0.05 for about 2.4× the time, and
  roughly matches or trails it slightly at p ≥ 0.06.
- **The small codes have no headroom.** On [[72,12,6]] and [[144,12,12]], BP+OSD is already close
  to the minimum-weight estimate, and the ensemble at best matches BP+OSD-CS40.
- **The larger codes have room for further improvement.** On [[288,12,18]] even the best decoder
  is still 2–4× above the minimum-weight estimate. Promising next steps:
  - more diverse groupings (for example, mixing `ell` values);
  - a native C++ ensemble that shares the OSD elimination between members;
  - running the lowest-cost selection over OSD-CS candidates from several members' soft outputs.

The BB construction, the exploration scripts, the headroom estimator (`mw_bound.py`) and the raw
data are in [`examples/bb`](examples/bb/README.md).

## License

MIT, see [LICENSE](LICENSE). This project is built on a fork of the `ldpc` package.
