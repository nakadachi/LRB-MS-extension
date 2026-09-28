## LDPC: Software for Decoding Classical and Quantum Codes

LDPC Version 2: A C++ rewrite of the `LDPCv1` package for decoding low density parity check checks.
Warning, whilst efforts have been made to provide backwards compatability with LDPCv1, the new version may introduce breaking changes.

## Documentation

The documentation for `LDPCv2` can be found [here](https://roffe.eu/software/ldpc)

## Installation

The easiest way to install the package is via pip. Python versions `>=3.10` are supported.

```pip install -U ldpc```

## Python - Installation from source

The C++ source code can be found in src_cpp. Python bindings are implemented using Cython and can be found in src/ldpc. To install the Python version of the repository follows the instructions below: 

- Download the repo.
- Navigate to the root.
- Pip install with `python>=3.10`.
Note: installation requires a `C` compiler. Eg. `gcc` on Linux or `clang` on Windows.

```
git clone git@github.com:quantumgizmos/ldpc_v2.git
cd ldpc
pip install -Ue .
```

## LDPCv1
If your package requires LDPCv1, this can be installed from PyPi as follows:

```pip install -U ldpc==0.1.60```

## New features

- A new C++ template class `GF2Sparse`. This is a more flexible implementation of the `mod2sparse` data structure used in the LDPCv1. This will make it much easier to expand the package.
- Serial schedules for the BP decoder.
- Run-time improvements for BP+OSD OSD-0. The decoder now implements the fast-syndrome OSD-0 implementation (https://arxiv.org/abs/1904.02703), where Gaussian elimination is terminated as soon as the syndrome becomes linearly dependent on the reduced columns.
- BP+LSD: Belief propagation plus localised statistics decoding. A parallel decoding algorithm that matches the perforance of BP+OSD. Note that the version implemented currenlty runs in serial. We are working on the parallel version! See our paper: https://arxiv.org/abs/2406.18655
- The union-find matching decoder (https://arxiv.org/abs/1709.06218). This is an implementation of the Delfosse-Nickerson union-find decoder that is suitable for decoding surface codes and other codes with "matchable" syndromes.
- The BeliefFind decoder. A decoder that first runs belief propagation, and falls back on union-find if if the BP decoder fails to converge as proposed by Oscar Higgott in https://arxiv.org/abs/2203.04948
- Flip and P-flip decoders as introduced by Thomas Scruby in https://arxiv.org/abs/2212.06985.
- Improved GF2 linear algebra routines (useful for computing code parameters)

## ToDos

`LDPCv2` is still a work in progress. Ongoing projects are listed below:
- Implement parallel version of BP+LSD algorithm using OpenMP.
- Improve support for parallel processing across the package.
- More decoders could be implemented (eg. small set-flip, https://arxiv.org/abs/1810.03681)
- Stabiliser inactivation BP (https://arxiv.org/abs/2205.06125)
- Generalised BP (https://arxiv.org/abs/2212.03214)
- Functions need to be properly documented (in progress)
- Further STIM integration
- More functionality for studying classical codes. Eg. support for received vector decoding and the AWGN noise channel.

## LRB-MS (generalized-check min-sum) Quickstart

`LrbmsDecoder` groups the rows of the parity-check matrix into *generalized checks*
(GCs) of `ell` rows and runs min-sum message passing between variable nodes and GCs.
Each GC returns max-log extrinsic LLRs over its local syndrome coset, approximated by a
candidate list built from its least-reliable basis (LRB-MS-`t`), or computed exactly with
a 2^`ell`-state syndrome trellis (`gc_method="trellis"`, reference baseline).
With one row per GC it reduces exactly to (normalised) min-sum BP.

```python
import numpy as np
from ldpc import LrbmsDecoder
from ldpc.lrbms_decoder import overlap_check_groups

decoder = LrbmsDecoder(
    H,                       # np.ndarray or scipy.sparse parity-check matrix
    error_rate=0.05,         # or error_channel=[...]
    check_groups=9,          # int ell (greedy overlap grouping) or explicit list of row lists
    lrbms_order=8,           # LRB-MS-t: order-2 re-encodings among the t least-reliable MRB positions
    gc_method="lrbms",       # or "trellis" (exact min-sum per GC, ell <= 22)
    schedule="serial",       # "parallel" (flooding) or "serial" (layered over GCs)
    max_iter=100,
    ms_scaling_factor=0.8,
)
correction = decoder.decode(syndrome)
decoder.converge, decoder.iterations, decoder.log_prob_ratios
```

A sinter wrapper is available as `ldpc.sinter_decoders.SinterLrbmsDecoder`.

### Benchmark: quantum Tanner codes

We benchmarked `LrbmsDecoder` against `ldpc`'s BP and BP+OSD on four Leverrier–Zémor
quantum Tanner codes, from [[576,32]] to [[2160,24]]. The noise is code-capacity bit flips.
Each generalized check is one Tanner-graph vertex's local tensor code (9 rows), which is
exactly what `overlap_check_groups(H, 9)` recovers.

![Quantum Tanner code benchmark](examples/qtanner/qtanner_benchmark.png)

| code, p | BP (min-sum) | BP+OSD-CS7 | LRB-MS-8 |
|---|---|---|---|
| [[576,32,≤16]] (C16), 0.05 | 4.3e-2 | 3.7e-2 | 1.4e-3 |
| [[864,22,≤18]] (SL(2,3)), 0.05 | 1.2e-2 | 8.4e-3 | 5e-5 |
| [[1080,24,≤18]] (D30), 0.06 | 5.7e-2 | 4.1e-2 | 1.4e-3 |
| [[2160,24]] (A5), 0.06 | 2.6e-2 | 9.7e-3 | 5e-5 |

On these codes, LRB-MS lowers the logical error rate by 20–200× compared with BP and
BP+OSD-CS7, and the rise in failures moves from p ≈ 0.03–0.04 to p ≈ 0.06–0.07. Decode time is
comparable to plain min-sum BP and lower than BP+OSD. OSD adds little in either case. The exact
per-group trellis matches LRB-MS-8 but runs 5–30× slower. The setup, full results, and scripts
to reproduce them are in [`examples/qtanner`](examples/qtanner/README.md).

## BP+LSD Quickstart

Usage of the new BP+LSD decoder from https://arxiv.org/abs/2406.18655. Similar to BP+OSD, the LSD decoder can be applied to any parity check matrix. We recommend you start with `lsd_order=0`. The speed/accuracy trade-off for higher order values can be explored from there. Example below:

```python
import numpy as np
import ldpc.codes
from ldpc.bplsd_decoder import BpLsdDecoder

H = ldpc.codes.hamming_code(5)

## The
bp_osd = BpLsdDecoder(
            H,
            error_rate = 0.1,
            bp_method = 'product_sum',
            max_iter = 2,
            schedule = 'serial',
            lsd_method = 'lsd_cs',
            lsd_order = 0
        )

syndrome = np.random.randint(size=H.shape[0], low=0, high=2).astype(np.uint8)

print(f"Syndrome: {syndrome}")
decoding = bp_osd.decode(syndrome)
print(f"Decoding: {decoding}")
decoding_syndrome = H@decoding % 2
print(f"Decoding syndrome: {decoding_syndrome}")
``` 

## Attribution

If you use this software in your research please cite as follows:

```
@software{Roffe_LDPC_Python_tools_2022,
author = {Roffe, Joschka},
title = {{LDPC: Python tools for low density parity check codes}},
url = {https://pypi.org/project/ldpc/},
year = {2022}
}
```

If you have used the BP+OSD class for quantum error correction, please also cite the following paper:

```
@article{roffe_decoding_2020,
   title={Decoding across the quantum low-density parity-check code landscape},
   volume={2},
   ISSN={2643-1564},
   url={http://dx.doi.org/10.1103/PhysRevResearch.2.043423},
   DOI={10.1103/physrevresearch.2.043423},
   number={4},
   journal={Physical Review Research},
   publisher={American Physical Society (APS)},
   author={Roffe, Joschka and White, David R. and Burton, Simon and Campbell, Earl},
   year={2020},
   month={Dec}
}
```
