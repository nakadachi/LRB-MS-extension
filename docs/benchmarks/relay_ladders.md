# Benchmark: (μ, α) relay ladders with the MBP4 + Qulid hybrid

A relay ladder retries a decode with new (μ, α) settings only when the previous attempt did not
converge, so the average cost barely changes. We tested ladders on BB [[288,12,18]] under
depolarizing code-capacity noise (p = total error probability, p/3 per Pauli). The decoder uses
greedy groups of ℓ = 6 rows per check type, I = 1000 iterations, LRB order 6 and a serial schedule.
Ladders were searched on a (μ, α) grid with μ in 0.45–1.0 and α in 0.6–1.25: chosen greedily on one
half of the base decoder's failures and scored on the other half. The final comparison uses fresh
shots (`examples/mbp_ladder.py`):

| [[288,12,18]] | base (0.75, 1) | μ-only ladder → (0.9, 1) → (0.55, 1) | (μ, α) ladder → (1.0, 0.9) → (0.9, 0.8) |
|---|---|---|---|
| p = 0.065, 600k shots | 107 fails (1.8e-4) | 18 (3.0e-5), ×5.9 | **9 (1.5e-5), ×11.9** |
| p = 0.075, 120k shots | 97 fails (8.1e-4) | 21 (1.75e-4), ×4.6 | **9 (7.5e-5), ×10.8** |

- **α is worth adjusting.** The best single retries all change α as well as μ. Retrying with
  (1.0, 0.9) alone rescued 73 of 85 (p = 0.075) and 38 of 43 (p = 0.065) held-out base failures,
  against 62 of 85 and 36 of 43 for (0.9, 1.0).
- **One retry can buy about an order of magnitude.** In the held-out search at p = 0.065, adding
  the leg (1.0, 0.9) cut failures 9.5×, from 1.27e-4 to 1.33e-5.
- **The floor is wrong convergence.** At p = 0.065, all 9 failures left after the (μ, α) ladder
  converged to a wrong correction, which a retry cannot fix. The ensemble's lightest-answer
  selection or OSD would have to address those.
- **No floor at p = 0.04.** The base leg had 0 failures in 600 000 shots at p = 0.04, and 0 in
  40 000 at p = 0.055, so there was no floor there to remove.
- **The Tanner code gains little in its waterfall.** On the C16 quantum Tanner code
  ([[576,32]], ℓ = 9 vertex groups, I = 40, LRB order 16), at p = 0.10 a 4-leg ladder gained only
  ×1.7: 1.25e-3 → 7.3e-4 on held-out shots. Almost a fifth of its failures there are wrong
  convergences, and the code had no failures at all in 4 000 shots at p ≤ 0.09. Its best first
  leg is μ = 0.75; μ = 0.45 fails 2% of shots at p = 0.08.

## BB [[144,12,12]]: tuning (μ, α) for relay decoding

Same decoder (ℓ = 6, I = 1000, LRB order 6), total depolarizing p = 0.06.

**The single leg: α trades wrong convergence against non-convergence.** Failures in 20k identical
shots, with wrong convergences in brackets (`examples/mbp_sweep.py`):

| μ \ α | 0.7 | 0.8 | 0.9 | 1.0 | 1.1 | 1.25 |
|---|---|---|---|---|---|---|
| 0.5 | 159 (89) | 115 (26) | 152 (10) | 323 (5) | 804 (1) | 3009 (0) |
| 0.75 | 282 (281) | 101 (101) | 55 (50) | 38 (15) | 57 (11) | 152 (6) |
| 0.9 | 391 (389) | 161 (161) | 68 (66) | 43 (37) | **31 (17)** | 37 (7) |
| 1.0 | 600 (576) | 176 (175) | 73 (72) | 47 (42) | **31 (20)** | 40 (11) |

With α < 1 the decoder converges readily but often to a wrong correction. With α > 1 it rarely
converges wrongly but more often fails to converge.

**Stopping ladders level off at about 1e-3.** Ladders were searched from conservative first legs,
scoring each retry by rescues minus new wrong convergences:
- (0.9, 1.25) → (0.9, 1.1) → (0.65, 1) → (1, 1.1): 9.2e-4;
- (1, 1.1) → (0.75, 1) → (0.9, 1): 1.0e-3;
- (0.75, 1.25) → (0.9, 1.1) → (0.9, 1.25): 1.0e-3.

The retries rescue nearly every non-converged shot, but a quarter to a third of those rescues
converge to a wrong correction. A stopping ladder cannot see that, so it cannot go much lower.

**Lightest of several legs does better.** Each shot runs k legs and keeps the valid output with the
fewest non-identity Paulis. Legs were chosen greedily on 20k training shots
(`examples/mbp_lightest_opt.py`) and compared on the same 80k fresh shots:

| [[144]], p = 0.06 | fails | gain | time / shot |
|---|---|---|---|
| single leg (0.75, 1) | 153 (1.9e-3) | — | 0.27 ms |
| tuned single leg (0.9, 1.1) | 125 | ×1.2 | 0.25 ms |
| lightest of 3, hand-picked: (0.75, 1), (1.0, 0.9), (0.9, 0.8) | 96 | ×1.6 | 0.78 ms |
| **lightest of 3, tuned: (0.9, 1.1), (0.75, 1.1), (0.9, 0.9)** | **78** | **×2.0** | 0.84 ms |
| lightest of 5 (either set) | 73 | ×2.1 | 1.4 ms |
| lightest of all 10 legs | 64 | ×2.4 | 2.9 ms |

An earlier 40k-shot run reported ×3.0 and ×4.1 for the hand-picked sets. Those counts were
optimistic; the table above supersedes them.

**This is close to the limit for [[144]].** Of the wrong convergences left after the ladder, about
half had the same weight as the true error or less, and those defeat any minimum-weight decoder.
The [code-capacity headroom estimate](bb_codes.md#how-much-can-be-gained) also put minimum-weight decoding only about 1.5× below
BP+OSD on this code. On [[288,12,18]], whose failures are mostly non-convergence, a stopping
ladder already gives 11–20× at no extra cost.

## Verifying the reported gross-code ladder

The original analysis reports a ×1.7–2.4 ladder gain for BB gross [[144,12,12]], capped by wrong
convergences. Its configuration:
- **Reference:** a single decode with ℓ = 6, LRB order δ = 6, (μ*, α, I) = (0.75, 1.0, 40).
- **Ladder:** the reference, then on non-convergence (0.90, 1.0, 150) → (0.75, 0.9, 150) →
  (0.60, 1.0, 150).

We reran exactly this under total depolarizing noise, with identical shots for all three
decoders. Legs can now carry their own iteration count in `examples/mbp_ladder.py`, e.g.
`"0.75,1.0,40;0.90,1.0,150"`.

| p | shots | reference (I = 40) | ladder | ladder gain | single decode, I = 1000 | ladder vs I = 1000 |
|---|---|---|---|---|---|---|
| 0.04 | 400k | 53 (1.3e-4) | 37 (34 wrong conv.) | ×1.4 | 43 | ×1.2 |
| 0.05 | 200k | 131 (6.6e-4) | 74 (66 wrong) | ×1.8 | 99 | ×1.3 |
| 0.06 | 120k | 301 (2.5e-3) | 161 (129 wrong) | ×1.9 | 220 | ×1.4 |
| 0.07 | 60k | 409 (6.8e-3) | 226 (176 wrong) | ×1.8 | 311 | ×1.4 |

- **The gain is confirmed:** ×1.4–1.9, at the lower end of the reported ×1.7–2.4.
- **About a third to a half of it is iteration budget.** A single decode at I = 1000 already
  improves on the 40-iteration reference by ×1.2–1.4. Against that, the ladder's own gain is
  ×1.2–1.4.
- **Wrong convergences cap it.** 80–92% of the ladder's remaining failures are wrong
  convergences, which no further rung can fix.
- **The ladder is cheap:** 0.14–0.28 ms per shot, about the reference's cost and less than the
  single I = 1000 decode at higher p. The rungs fire on only 0.006–0.5% of shots.

## The screenshot setups

We reran three relay-ladder setups from the original analysis notes with this decoder. Everything
here uses **total depolarizing noise**: p is the total error probability per qubit, split p/3 per
Pauli. This is the convention for all results in this repository. Every ladder sees the same shots.

| setup | decoder | base leg | μ-only ladder | (μ, α) ladder |
|---|---|---|---|---|
| A: [[288,12,18]] floor, p = 0.06, 600k shots | ℓ = 6, I = 1000, LRB order 1 | (0.75, 1): 20 fails (3.3e-5) | → (0.9, 1) → (0.55, 1): 3 (5.0e-6), ×6.7 | → (1.0, 0.9) → (0.9, 0.8): **1 (1.7e-6), ×20** |
| B: [[288,12,18]] near threshold, p = 0.075, 120k shots | same | 67 fails (5.6e-4) | 11 (9.2e-5), ×6.1 | **6 (5.0e-5), ×11.2** |
| C: [[432,20]] quantum Tanner, p = 0.03, 1.2M shots | ℓ = 9 (vertex groups), I = 40, LRB order 16 | (0.45, 1): 95 fails (7.9e-5) | → (0.60, 1): **0** (≥ ×95) | — |

- **A (floor):** at p = 0.04 this decoder has no failures in 600k shots, with LRB order 6 or 1, so
  the floor is probed at p = 0.06. There the base leg's 20 failures in 600k match the 19 that the
  notes report for their floor run. The μ-only ladder leaves 3 (notes: 1). The (μ, α) ladder leaves
  1, a wrong convergence, so a retry cannot remove it.
- **B (near threshold):** the notes report the μ-only ladder at ×4.1; we measure ×6.1. Adding α to
  the ladder roughly doubles the gain. The notes' base rate (2.0e-3) corresponds to p ≈ 0.085 here:
  2.3e-3 base, and 6.75e-4 after the μ-only ladder (notes: 4.9e-4).
- **C:** the original [[432,20,22]] code was not available. We used a stand-in built on A4 with
  [6,3,3] local codes (`examples/qtanner/search_432.py`). It has k = 20 but distance ≤ 16
  (information-set bound): no order-12 instance we found reached d = 22. As in the notes, the
  second leg removed every base failure. However, a single leg with μ = 0.75 also had 0 failures
  in the same 1.2M shots. So on this code the floor at μ = 0.45 comes from the first leg's μ being
  too small, and the retry mainly moves back towards a better μ.

The notes' p values do not map onto total depolarizing p consistently. Their floor run matches our
p = 0.06, which is 1.5× their 0.04; their near-threshold run matches p ≈ 0.085, which is 1.13× their
0.075. So their two runs probably used different noise settings. `examples/mbp_ladder.py` can
sample other conventions through the environment variable `MBP_CONV`, for comparison only:
- `perpauli`: p per Pauli;
- `marginal`: the X and Z parts each flip with probability p;
- `indep`: independent X and Z flips.

The default, `total`, is the one used throughout.

---
[← back to the README](../../README.md)
