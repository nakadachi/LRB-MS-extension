# Comparison with related generalized-check decoders

Three recent papers also decode quantum Tanner codes by treating each vertex's local code as one
generalized check. We implemented all three in this package and compared them with LRB-MS on
the same codes, the same noise and the same error samples.

| paper | idea | where it lives here |
|---|---|---|
| Mostad, Rosnes and Lin, [arXiv:2603.05486](https://arxiv.org/abs/2603.05486) | MBP4 with each generalized check decoded by exact MAP (BCJR on a syndrome trellis); hybrid with plain MBP4 first and OSD-1 last | `gc_method="map"`, `GmbpDecoder`, `union_overlap_check_groups` |
| Rapp, Médard, Tang and Duffy, [arXiv:2603.18318](https://arxiv.org/abs/2603.18318) | each generalized check decoded by soft-output GRAND: an ORBGRAND list plus the probability that the true pattern is not in it | `gc_method="sogrand"` |
| Xiao, Shi, Huang, Wang and Wang, [arXiv:2605.17796](https://arxiv.org/abs/2605.17796) | LEAD: BP-LSD on every local code, local estimates averaged into a prior, then global BP-OSD (one pass) | `LeadDecoder` |

LRB-MS sits between the first two. Like SOGRAND it scores a short list of local patterns
instead of the whole coset; unlike SOGRAND the list comes from a least-reliable-basis
re-encoding and the output is max-log. Exact MAP is the reference for both.

## Implementation notes

- **MAP check (Mostad et al., Alg. 2):** forward/backward over the 2^ℓ-state syndrome trellis
  in the probability domain. Each step keeps unit total mass, so no rescaling is needed. The
  output is extrinsic. With one row per check it is the box-plus rule, so single-row MAP equals
  `ldpc`'s product-sum BP (tested).
- **GMBP4 (Mostad et al., Alg. 3):** their variable rule scales check outputs by 1/a in both
  the posterior and the outgoing messages, with a flooding schedule. That is
  `MbpLrbmsDecoder(gc_method="map", mu=1/a, alpha=1, schedule="parallel")`. `GmbpDecoder` adds
  their hybrid: 6 iterations of MBP4 with box-plus single-row checks, then 6 iterations with
  generalized checks, then OSD-1. OSD-1 is `ldpc`'s combination sweep of order 1 (order 0 is
  OSD-0 in `ldpc`). It runs on each CSS half that still violates its syndrome, from the
  marginals of the quaternary posteriors. a = 1.6 for quantum Tanner codes and 1.5 for BB.
- **SOGRAND (Rapp et al.):** written from the papers' description. Their reference code is
  under a non-commercial licence and was only read, not copied. Details:
  - patterns are queried in 1-line ORBGRAND order with the automatic intercept;
  - the search stops at list size 4, or once the estimated probability that the true pattern
    is not in the list (2^-rank × unqueried mass) is below 10^-5;
  - the not-in-list mass is spread over the bits by their input marginals, and the output is
    extrinsic;
  - the binary variant decodes X and Z independently with the marginal rate 2p/3; the X/Z-aware
    variant runs inside MBP4 with a = 1, which is their correlation-aware message passing;
  - 200 iterations, no normalisation.
  Within one pattern weight the query order is lexicographic, where the reference uses its
  "landslide" order. It is tested against an independent Python transcription and, with an
  unlimited list, against exact MAP.
- **LEAD (Xiao et al., Alg. 1):** local BP-LSD (`lsd_cs`, order 3, min-sum, iterations = local
  length) on each vertex's local code. Bits the local decision flips are raised to at least 0.5.
  The local estimates are averaged and scaled (α = 1), and a global BP-OSD (`osd_cs`, order 3,
  min-sum, iterations = n) runs from that prior. X and Z are decoded independently at 2p/3.
  The paper does not say how local views with a zero syndrome enter the average (`ldpc`'s
  BP-LSD skips them). We use the channel prior.
- **LRB-MS (ours):** `MbpLrbmsDecoder` with one generalized check per vertex, LRB order 8,
  serial schedule, α = 1, at most 100 iterations, no OSD. μ was chosen per code on separate
  error samples: 0.95 for [[250,10,15]], whose local codes are high-rate (6 checks on 25 bits),
  and 0.75 for the other two.

All decoders see the same depolarizing errors (p total, p/3 per Pauli). A shot fails if a
decoded half misses its syndrome or the residual flips a logical. Decode time is the mean wall
time per shot on one core (four workers in parallel). Everything is C++ except two Python
wrappers: LEAD's loop over local decoders and GMBP4's stage switch. The scripts are in
[`examples/related_work`](../../examples/related_work).

## Validation against the papers

Each reimplementation was checked against numbers from its own paper before comparing. For
SOGRAND they come from the result files in the authors' repository; for the others they are
read off the papers' figures, so they are approximate.

| paper, code | decoder | p | paper | ours |
|---|---|---|---|---|
| Rapp et al., [[250,10,15]] | SOGRAND (binary, 200 it.) | 0.0392 | 3.3e-3 | 2.8e-3 (50/17 600) |
| | SOGRAND+XZ (200 it.) | 0.0392 | 9.1e-5 | 9.0e-5 (9/100 000) |
| | BP, sum-product, normalised 0.8 | 0.0392 | ≈ 1.6e-2 (between 0.0373 and 0.0535) | 2.1e-2 (30/1 400) |
| Mostad et al., [[432,16]] (Fig. 3) | MBP4+OSD-1 (r = 1) | 0.05 / 0.08 | ≈ 1e-2 / ≈ 0.35 | 1.0e-2 / 0.33 |
| | GMBP4+OSD-1, r = 12 (full grouping) | 0.08 | ≈ 7e-4 | 0 / 3 000 |
| | GMBP4+OSD-1, r = 3 (rows sharing an H_B row) | 0.05 / 0.08 | ≈ 4e-4 / ≈ 0.1 | 3.3e-4 / 6.6e-2 |
| | GMBP4+OSD-1, r = 4 (rows sharing an H_A row) | 0.08 | ≈ 0.1 | 3.6e-2 |
| Xiao et al., [[36,8,3]] (Fig. 4a) | BP-OSD (min-sum, OSD-CS3) | 0.01 / 0.02 / 0.03 | ≈ 9e-3 / 3.6e-2 / 7.6e-2 | 1.1e-2 / 3.8e-2 / 8.2e-2 |
| | LEAD (BL–BO, α = 1) | 0.01 / 0.02 / 0.03 | ≈ 6.5e-3 / 2.6e-2 / 6.0e-2 | 8.8e-3 / 3.4e-2 / 8.1e-2 |

- **SOGRAND** reproduces the published numbers closely: the X/Z-aware result is within 1% of
  the paper's at p = 0.0392.
- **GMBP4** reproduces Mostad et al.'s baseline and partial groupings. Our r = 3 and r = 4
  curves sit at or slightly below theirs.
- **LEAD:** our BP-OSD baseline matches the paper's, but our LEAD gains less: 1.0–1.25× over
  BP-OSD instead of about 1.4×. Neither alternative for zero-syndrome views (estimate 0, or
  leave them out of the average) closes the gap. Leaving them out is much worse. The remaining
  difference is in details the paper does not give, so the LEAD numbers below are those of our
  reimplementation.

## Head-to-head

_Head-to-head results are being generated (`examples/related_work/run_all.sh`) and will be added here._

---
[← back to the README](../../README.md)
