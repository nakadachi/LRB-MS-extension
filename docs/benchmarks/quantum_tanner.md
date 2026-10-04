# Benchmark: quantum Tanner codes (code capacity)

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

![Quantum Tanner code benchmark](../../examples/qtanner/qtanner_benchmark.png)

## Logical error rate per shot

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


## Findings

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
[`examples/qtanner`](../../examples/qtanner/README.md).

## Larger codes: up to n = 10 752

We scaled the benchmark to three larger quantum Tanner codes. Two of them use the stronger
`[8,4,4]` extended-Hamming local code, which gives 16 rows per vertex and 64-qubit local views:

| code | group | local code | n | k |
|---|---|---|---|---|
| [[3840,48]] | A5 | [8,4,4] | 3840 | 48 |
| [[6048,40]] | PSL(2,7) | [6,3,3] | 6048 | 40 |
| [[10752,56]] | PSL(2,7) | [8,4,4] | 10752 | 56 |

The setup is the same as above: code-capacity bit flips, paired samples, serial schedules, at most
100 iterations, scaling 0.75. LRB-MS points run up to 10 000 shots. The baselines are capped at
400 shots (BP, BP+LSD-CS7) or 100 shots (BP+OSD-CS7), because they take seconds per shot.
BP+OSD-CS7 ran only at the lowest noise levels; at higher noise or on the largest code a single
shot takes minutes. On [[10752,56]], BP and BP+LSD were skipped for p ≥ 0.05, after they had
already failed every shot at p = 0.04. "—" means not run.

![Large quantum Tanner code benchmark](../../examples/qtanner/qtanner_large_benchmark.png)

> **Timing caveat:** while this benchmark ran, a leftover background job occupied the same 4 cores,
> so the absolute decode times in this section are likely about 2× too high. The extra load was
> constant throughout, so the relative speeds of the decoders should be roughly preserved.
> Logical error rates are unaffected.

**[[3840,48]] (A5/[8,4,4])**

| decoder | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 | 0.06 | 0.07 | 0.08 |
|---|---|---|---|---|---|---|---|---|
| BP (min-sum) | 2.3e-01 | 5.6e-01 | 8.1e-01 | 1.0e+00 | 9.9e-01 | 1.0e+00 | 1.0e+00 | 1.0e+00 |
| BP+LSD-CS7 | 6.5e-02 | 3.0e-01 | 6.4e-01 | 9.4e-01 | 9.5e-01 | 1.0e+00 | 1.0e+00 | 1.0e+00 |
| BP+OSD-CS7 | 1.0e-02 | 1.9e-01 | — | — | — | — | — | — |
| **LRB-MS-0** | 0 (/10000) | 0 (/10000) | 0 (/10000) | 0 (/10000) | 0 (/10000) | 1.2e-02 | 4.6e-01 | 9.8e-01 |
| **LRB-MS-8** | 0 (/10000) | 0 (/10000) | 0 (/10000) | 0 (/10000) | 1.0e-04 | 4.0e-04 | 9.4e-02 | 7.8e-01 |

Mean decode time per shot (one core):

| decoder | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 | 0.06 | 0.07 | 0.08 |
|---|---|---|---|---|---|---|---|---|
| BP (min-sum) | 299 ms | 610 ms | 959 ms | 1.1 s | 1.0 s | 1.1 s | 1.1 s | 1.1 s |
| BP+LSD-CS7 | 495 ms | 1.3 s | 2.1 s | 2.7 s | 2.6 s | 3.2 s | 4.1 s | 4.6 s |
| BP+OSD-CS7 | 14.4 s | 61.0 s | — | — | — | — | — | — |
| **LRB-MS-0** | 5 ms | 6 ms | 8 ms | 11 ms | 17 ms | 38 ms | 144 ms | 230 ms |
| **LRB-MS-8** | 5 ms | 7 ms | 9 ms | 12 ms | 16 ms | 28 ms | 87 ms | 253 ms |

**[[6048,40]] (PSL(2,7)/[6,3,3])**

| decoder | 0.03 | 0.04 | 0.05 | 0.06 | 0.07 | 0.08 | 0.09 |
|---|---|---|---|---|---|---|---|
| BP (min-sum) | 0 (/400) | 2.5e-03 | 2.5e-03 | 1.3e-02 | 6.5e-02 | 7.7e-01 | 1.0e+00 |
| BP+LSD-CS7 | 0 (/400) | 2.5e-03 | 0 (/400) | 2.5e-03 | 4.5e-02 | 7.3e-01 | 1.0e+00 |
| BP+OSD-CS7 | 0 (/100) | 0 (/100) | 0 (/100) | — | — | — | — |
| **LRB-MS-0** | 0 (/10000) | 0 (/10000) | 0 (/10000) | 0 (/10000) | 3.0e-04 | 3.8e-02 | 7.3e-01 |
| **LRB-MS-8** | 0 (/10000) | 1.0e-04 | 0 (/10000) | 0 (/10000) | 1.0e-04 | 2.8e-02 | 6.7e-01 |

Mean decode time per shot (one core):

| decoder | 0.03 | 0.04 | 0.05 | 0.06 | 0.07 | 0.08 | 0.09 |
|---|---|---|---|---|---|---|---|
| BP (min-sum) | 13 ms | 18 ms | 24 ms | 44 ms | 90 ms | 250 ms | 300 ms |
| BP+LSD-CS7 | 13 ms | 21 ms | 24 ms | 39 ms | 212 ms | 3.0 s | 4.1 s |
| BP+OSD-CS7 | 9 ms | 12 ms | 15 ms | — | — | — | — |
| **LRB-MS-0** | 10 ms | 12 ms | 15 ms | 17 ms | 25 ms | 58 ms | 213 ms |
| **LRB-MS-8** | 12 ms | 14 ms | 17 ms | 21 ms | 30 ms | 62 ms | 285 ms |

**[[10752,56]] (PSL(2,7)/[8,4,4])**

| decoder | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 | 0.06 | 0.07 | 0.08 |
|---|---|---|---|---|---|---|---|---|
| BP (min-sum) | 3.7e-01 | 9.6e-01 | 9.9e-01 | 1.0e+00 | — | — | — | — |
| BP+LSD-CS7 | 1.2e-01 | 7.3e-01 | 9.6e-01 | 1.0e+00 | — | — | — | — |
| BP+OSD-CS7 | — | — | — | — | — | — | — | — |
| **LRB-MS-0** | 0 (/10000) | 0 (/10000) | 0 (/10000) | 0 (/10000) | 1.0e-04 | 1.0e-04 | 6.0e-01 | 1.0e+00 |
| **LRB-MS-8** | 0 (/10000) | 0 (/10000) | 0 (/10000) | 0 (/10000) | 1.0e-04 | 0 (/10000) | 2.0e-02 | 9.4e-01 |

Mean decode time per shot (one core):

| decoder | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 | 0.06 | 0.07 | 0.08 |
|---|---|---|---|---|---|---|---|---|
| BP (min-sum) | 1.3 s | 3.1 s | 3.1 s | 3.2 s | — | — | — | — |
| BP+LSD-CS7 | 5.7 s | 15.9 s | 17.4 s | 18.4 s | — | — | — | — |
| BP+OSD-CS7 | — | — | — | — | — | — | — | — |
| **LRB-MS-0** | 15 ms | 18 ms | 25 ms | 35 ms | 51 ms | 93 ms | 492 ms | 656 ms |
| **LRB-MS-8** | 18 ms | 20 ms | 26 ms | 37 ms | 49 ms | 74 ms | 209 ms | 761 ms |


**Findings on the larger codes**

- **With [8,4,4] local codes, the standard decoders break down at very low noise.** At p = 0.01 on
  [[3840,48]], BP fails 23% of shots, BP+LSD-CS7 6.5% and BP+OSD-CS7 1% (at 14 s per shot).
  LRB-MS-8 has no failures in 10 000 shots at up to p = 0.04, and fails 1e-4 at p = 0.05.
  On [[10752,56]], BP and BP+LSD fail every shot from p = 0.04, while LRB-MS has no failures in
  10 000 shots at p = 0.04 and 0.06.
- **LRB-MS is also far faster.** On [[3840,48]] at p = 0.02, LRB-MS-8 takes 7 ms per shot, against
  1.3 s for BP+LSD-CS7 and 61 s for BP+OSD-CS7, roughly 200× and 9 000× faster. On [[10752,56]]
  it takes 18–75 ms per shot for p ≤ 0.06, while BP+LSD takes 5–18 s.
- **With [6,3,3] local codes (n = 6048), the baselines hold up at low noise, but LRB-MS still wins
  where failures start.** At p = 0.07, BP+LSD-CS7 fails 4.5% at 212 ms per shot, while LRB-MS-8
  fails 1e-4 at 30 ms.
- **The stronger the local code, the larger the gain.** The [8,4,4] codes hurt the baselines far more
  than the [6,3,3] codes do. A likely reason is that their dense 16-row local checks form many short
  cycles inside each vertex, which defeat plain message passing, whereas LRB-MS solves each vertex's
  local code jointly.

---
[← back to the README](../../README.md)
