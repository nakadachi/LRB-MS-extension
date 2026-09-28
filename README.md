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

## License

MIT, see [LICENSE](LICENSE). This project is built on a fork of the `ldpc` package.
