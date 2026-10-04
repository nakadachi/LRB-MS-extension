# Using the decoders

Installation and a minimal example are in the [README](../README.md#quickstart).

## Choosing the generalized checks

- `check_groups=None` gives one row per GC, which is plain min-sum BP.
- `check_groups=ell` (an integer) uses `overlap_check_groups(H, ell)`. Each group is grown
  greedily from rows that share the most columns.
- `consecutive_check_groups(m, ell)` makes blocks of consecutive rows. Use it when neighbouring
  rows already form a component code (GLDPC or Tanner-type constructions).
- You can also pass an explicit list of row lists. Every nonzero row must be covered.
- Groups should partition the rows. Putting a row in two groups counts its evidence twice and
  makes decoding much worse.

## Ensemble over groupings

`LrbmsEnsembleDecoder` runs LRB-MS once per check grouping and returns the syndrome-valid output
with the lowest channel cost. Different groupings put the GC boundaries in different places, so
their failures are only weakly correlated. This helps most on codes without a natural local-code
structure, such as bivariate bicycle codes (see [the BB benchmark](benchmarks/bb_codes.md)).

```python
from ldpc.lrbms_decoder import LrbmsEnsembleDecoder

decoder = LrbmsEnsembleDecoder(
    H, error_rate=0.05,
    ell=8, num_groupings=8,  # greedy groupings grown from permuted row orders
                             # (or pass groupings=[...] explicitly)
    stop="escalate",         # "escalate": accept the first grouping if it converges, else run all (recommended)
                             # "all": always run every grouping; "first": stop at the first converged one
    osd_members="all",       # OSD fallback on every grouping ("last": only on the last one)
    osd_method="osd_cs", osd_order=7,
    max_iter=100, ms_scaling_factor=0.75, schedule="serial",
)
correction = decoder.decode(syndrome)
decoder.converge, decoder.member
```

**Choosing the mode.** The ensemble's gain comes almost entirely from shots where the first grouping
does not converge. In those shots, OSD on several groupings' soft outputs gives several candidates,
and the lightest one is usually right. Choosing among groupings that all converged gains nothing:
"all", with OSD only when nothing converged, is no better than "first". Agreement-based and
weight-based stopping rules also gained nothing. So `stop="escalate"` runs the extra groupings only
when the first one fails. On BB codes at code capacity, with 8 groupings, it matched "all" in accuracy
at a fraction of the cost:

| code, p | all | **escalate** | first |
|---|---|---|---|
| [[288,12,18]], 0.06 | 1.30e-2, 5.6 ms | **1.32e-2, 2.9 ms** | 1.95e-2, 1.9 ms |
| [[756,16,≤34]], 0.065 | 4.0e-3, 15.9 ms | **4.0e-3, 5.9 ms** | 6.5e-3, 3.3 ms |

Escalate costs about 3–4× a single LRB-MS decode, instead of about 8× for "all". The groupings
are independent, so running them in parallel would cut the latency further. Capping `max_iter`
per grouping helps on some codes: on [[288,12,18]], 10 iterations kept the accuracy and cut the
"all" time by about 37%. It hurts on others: on [[756,16,≤34]] more groupings fall back to OSD,
and the time grows. The script is `examples/bb/ensemble_policies.py`. The BB benchmark tables were measured with "all" and "first".

## MBP4 + LRB-MS hybrid (depolarizing noise)

`MbpLrbmsDecoder` decodes the X and Z parts of a CSS-code error jointly. Each qubit keeps
quaternary log-ratios Γᵂ = ln P(I)/P(W), for W = X, Y, Z, updated with the MBP4 rule of
Kuo and Lai. The checks are LRB-MS generalized checks, grouped within one check type: a Z-type
group sees only the x-part of each qubit's error, and an X-type group only the z-part. The
decoder has two scaling knobs:

- **μ (`mu`)** scales each check group's output (min-sum normalisation at the check nodes).
- **α (`alpha`)** weights incoming check messages by 1/α in each qubit's posterior. The message
  back to a check subtracts that check's previous output without the 1/α factor, which is
  MBP's memory / inhibition effect for α < 1.

With p_Y = p_Z = 0 and α = 1, the decoder reproduces binary LRB-MS exactly (tested).

```python
from ldpc.lrbms_decoder import MbpLrbmsDecoder

dec = MbpLrbmsDecoder(hx, hz, error_rate=0.06,      # depolarizing p (p/3 per Pauli); or channel=(px, py, pz)
                      x_groups=6, z_groups=6,       # groupings of the rows of hx / hz
                      max_iter=1000, mu=0.75, alpha=1.0, lrbms_order=6, schedule="serial")
ex, ez = dec.decode(hx @ ez_true % 2, hz @ ex_true % 2)
dec.converge
dec.mu, dec.alpha = 1.0, 0.9                        # change the knobs between decodes (e.g. relay retries)
```

## Sinter

For stim/sinter workflows use `ldpc.sinter_decoders.SinterLrbmsDecoder`.

## Tests

```bash
pytest python_test/test_lrbms.py
```

The tests check that:
- `ell = 1` reproduces min-sum BP exactly.
- The trellis matches brute-force max-log.
- LRB-MS is exact whenever its candidate list covers the coset.
- The OSD fallback works.
- The ensemble returns the cheapest valid member output, and `stop="escalate"` matches `"all"`
  whenever the first grouping fails.
- The MBP4 + LRB-MS hybrid reproduces binary LRB-MS for pure X noise with α = 1.

---
[← back to the README](../README.md)
