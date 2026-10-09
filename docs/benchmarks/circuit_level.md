# Benchmark: circuit-level noise, BB [[144,12,12]]

We also decoded BB [[144,12,12]] under circuit-level noise. This is a Z-basis memory experiment with
12 rounds of syndrome extraction, built by `ldpc.ckt_noise.make_css_code_memory_circuit`.
Every gate, idle step, reset and measurement fails with probability `p`. Only Z detectors are kept,
and the whole 12-round history is decoded at once. The decoding matrix is the stim detector error
model: 936 detectors × 8784 error mechanisms, with per-column priors.

Qulid groups detectors rather than code checks. We tried four groupings:
- the same stabilizer in consecutive rounds;
- spatial groups of stabilizers, within one round or spanning two rounds;
- greedy overlap on the detector matrix;
- ensembles of greedy groupings grown from permuted row orders (`ell = 8`).

Every decoder sees the same stim samples, uses a serial schedule with at most 100 iterations, and
uses min-sum scaling 0.75. The single Qulid decoder uses 0.9, which was equally accurate and faster.
A point stops at 100 failures or 4 000 shots, or 2 000 shots for the run-all ensemble.

![Circuit-level BB benchmark](../../examples/bb_circuit/bb_circuit_benchmark.png)

> **Timing caveat:** while this benchmark ran, a leftover background job occupied the same 4 cores,
> so the absolute decode times in this section are likely about 2× too high. The extra load was
> constant throughout, so the relative speeds of the decoders should be roughly preserved.
> Logical error rates are unaffected.

Logical error rate per shot (12 rounds), with mean decode time per shot on one core:

| decoder | p = 0.002 | p = 0.0025 | p = 0.003 |
|---|---|---|---|
| BP+OSD-CS7 | 1.5e-3 (6/4000), 176 ms | 1.1e-2, 367 ms | 3.4e-2, 644 ms |
| BP+OSD-CS40 | 1.8e-3 (7/4000), 179 ms | 1.05e-2, 296 ms | 3.4e-2, 656 ms |
| Qulid greedy ell=8 + OSD-CS7 | 6.0e-3, 55 ms | 1.5e-2, 97 ms | 4.6e-2, 275 ms |
| **Ensemble ×8, first, OSD on last** | 1.5e-3 (6/4000), 135 ms | 7.5e-3, 305 ms | 3.5e-2, 776 ms |
| **Ensemble ×8, all, OSD on all** | 0 (/2000), 1.9 s | 5.0e-3 (10/2000), 3.7 s | 1.4e-2, 5.4 s |

**Findings at circuit level**

- **A single Qulid decoder is not better than BP+OSD here.** No detector grouping beat plain BP
  by much on its own: all were at 30–40% failures at p = 0.003. With an OSD fallback, Qulid is
  1.4–4× less accurate than BP+OSD-CS7, but 2–4× faster.
- **A higher OSD order does not help the baseline.** BP+OSD-CS40 matches BP+OSD-CS7 at every point.
- **The ensemble does help.** At p = 0.0025, "ensemble ×8, first" fails 1.4× less often than BP+OSD
  at the same decode time. "Ensemble ×8, all" fails 2.1–2.4× less often at p = 0.0025–0.003, but costs
  about 8–12× the time. At p = 0.002 the counts are too small to separate the decoders.
- **This matches the code-capacity BB result.** BB detectors do not form strong local codes, so the
  gain comes from trying several groupings, not from the per-group LRB update. A native C++
  ensemble, with the groupings run in parallel, is the natural next step to reduce its cost.

The circuit construction, detector groupings and benchmark script are in
[`examples/bb_circuit`](../../examples/bb_circuit).

---
[← back to the README](../../README.md)
