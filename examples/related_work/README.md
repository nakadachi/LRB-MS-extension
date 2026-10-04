# Comparison with related generalized-check decoders

Scripts for comparing LRB-MS with three related decoders for quantum Tanner codes, all built
in this package:

| paper | decoder here |
|---|---|
| Mostad, Rosnes and Lin, arXiv:2603.05486 | `GmbpDecoder` (MBP4, then MBP4 with exact MAP generalized checks, then OSD-1); `gc_method="map"` |
| Rapp, Médard, Tang and Duffy, arXiv:2603.18318 | `gc_method="sogrand"` in `LrbmsDecoder` (binary) and `MbpLrbmsDecoder` (X/Z-correlated) |
| Xiao, Shi, Huang, Wang and Wang, arXiv:2605.17796 | `LeadDecoder` (BP-LSD per local code, averaged prior, global BP-OSD) |

The results are in [docs/benchmarks/related_work.md](../../docs/benchmarks/related_work.md).

## Code instances

The papers' codes come from two public repositories without a license, so they are not copied
here. `./fetch_codes.sh` clones them into `external/` (ignored by git):

- [RebKatRad/qTanner](https://github.com/RebKatRad/qTanner): [[36,8,3]], [[200,10,10]],
  [[250,10,15]] (Radebold, Bartlett and Doherty), used by the SOGRAND and LEAD papers.
- [lilyxy/Check-weight-constrained-quantum-codes](https://github.com/lilyxy/Check-weight-constrained-quantum-codes):
  the [[432,16]] quantum Tanner code ([6,3] × [8,4] local codes, 12 checks per vertex) used by
  Mostad et al., plus their LP, HGP and BB codes.

`codes.py` loads them (labels `qt36`, `qt200`, `qt250`, `qt432`, `lp416`, `hgp377`, `bb144`)
and our C16 [[576,32]] code (`qtC16`). Their stabilizer matrices list the checks vertex by
vertex, so the full grouping (one generalized check per Tanner-graph vertex) is consecutive
blocks of rows.

## Running

```bash
./fetch_codes.sh
python compare.py --code qt250 --decoders decoders_qt250.json --ps 0.0392,0.0518 \
    --max-shots 100000 --max-fails 50 --seed-salt test --out results_qt250.json
./run_all.sh            # every code and noise level of the write-up (about 2 hours on 4 cores)
python plot_results.py  # figure and tables for the write-up
```

All decoders see the same depolarizing errors (p total, p/3 per Pauli) at each (code, p). A
shot fails if a decoded half misses its syndrome or the residual flips a logical. Decode time
is measured per shot on one core. The `decoders_*.json` files hold each decoder's settings:
the papers' own for theirs, and the scaling for LRB-MS chosen on separate samples
(`--seed-salt tune`).

The other `decoders_*.json` files are the validation runs against the papers' reported
numbers (`decoders_sogrand.json`, `decoders_mostad.json`, `decoders_lead.json`).
