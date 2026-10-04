"""Code instances used by the related-work papers, loaded from their public repositories.

The repositories carry no license, so their files are not copied into this one. Fetch them
with ``fetch_codes.sh`` (or set QTANNER_DATA / QLDPC_DATABASE to existing checkouts):

- Radebold, Bartlett and Doherty, "Explicit instances of quantum Tanner codes"
  (github.com/RebKatRad/qTanner): [[36,8,3]], [[200,10,10]], [[250,10,15]], ... Rows of the
  stabilizer matrices are ordered vertex by vertex.
- Wang, Liu, Li, Kubica and Gu, "Check-weight-constrained quantum codes"
  (github.com/lilyxy/Check-weight-constrained-quantum-codes): [[432,16,28]] (the [[432,16]]
  code of Mostad et al.), LP [[416,18,22]], HGP [[377,25,5]], BB codes.
"""

import os
import sys
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qtanner"))
sys.path.insert(0, str(HERE.parent / "bb"))

EXTERNAL = HERE / "external"
QTANNER_DATA = Path(os.environ.get("QTANNER_DATA", EXTERNAL / "qtanner" / "data"))
QLDPC_DATABASE = Path(
    os.environ.get("QLDPC_DATABASE", EXTERNAL / "check-weight-constrained-quantum-codes")
)

# label: (source, location, rows per vertex or None)
CODES = {
    "qt36": ("radebold", "[[36_8_3]]", 2),
    "qt200": ("radebold", "[[200_10_10]]", 6),
    "qt250": ("radebold", "[[250_10_15]]", 6),
    "qt432": (
        "database",
        "quantum_tanner/G9-1_A6-3_T5c4d5f54d04e_B8-4_Te71519c717c8_rep1_perm12",
        12,
    ),
    "lp416": ("database", "lifted_product/pk_code_169_n416_k18_d22", None),
    "hgp377": ("database", "hgp/hgp_16_4_6_n377_k25_d5", None),
    "bb144": ("database", "bivariate_bicycle/bb_code_12_6_n144_k12_d12", None),
    "qtC16": ("ours", "C16", 9),
}


class CssCode:
    def __init__(self, name, hx, hz, rows_per_vertex=None):
        self.name = name
        self.hx = scipy.sparse.csr_matrix(hx).astype(np.uint8)
        self.hz = scipy.sparse.csr_matrix(hz).astype(np.uint8)
        self.n = self.hx.shape[1]
        self.rows_per_vertex = rows_per_vertex
        from qtanner_codes import css_logicals

        self.lz = css_logicals(self.hx, self.hz).astype(np.int64)  # detects x-part residuals
        self.lx = css_logicals(self.hz, self.hx).astype(np.int64)
        self.k = self.lz.shape[0]

    def groups(self, spec, basis):
        """Check groups of hx (basis 'x') or hz (basis 'z') for a grouping spec.

        None or 'rows': one check per row; 'vertex': one group per Tanner-graph vertex;
        ('stride', s): rows of a vertex whose index differs by multiples of s (a partial
        grouping, e.g. the rows sharing a row of H_B); ('blocks', b): consecutive blocks of b
        rows; int r: Algorithm 1 of Mostad et al.;
        ('overlap', ell): overlap_check_groups.
        """
        from ldpc.lrbms_decoder import overlap_check_groups, union_overlap_check_groups

        pcm = self.hx if basis == "x" else self.hz
        rows = pcm.shape[0]
        if spec is None or spec == "rows":
            return None
        if spec == "vertex":
            block = self.rows_per_vertex
            return [list(range(first, first + block)) for first in range(0, rows, block)]
        if isinstance(spec, (list, tuple)) and spec[0] == "stride":
            block, stride = self.rows_per_vertex, spec[1]
            return [
                list(range(first + offset, first + block, stride))
                for first in range(0, rows, block)
                for offset in range(stride)
            ]
        if isinstance(spec, (list, tuple)) and spec[0] == "blocks":
            return [list(range(first, first + spec[1])) for first in range(0, rows, spec[1])]
        if isinstance(spec, (list, tuple)) and spec[0] == "overlap":
            return overlap_check_groups(pcm, spec[1])
        return union_overlap_check_groups(pcm, int(spec), seed=0)


def _read_tsv(path):
    return np.loadtxt(path, dtype=np.uint8, ndmin=2)


def load(label):
    source, location, rows_per_vertex = CODES[label]
    if source == "radebold":
        folder = QTANNER_DATA / location
        hx, hz = _read_tsv(folder / "X_stabs"), _read_tsv(folder / "Z_stabs")
    elif source == "database":
        prefix = QLDPC_DATABASE / "codes" / location
        hx = scipy.io.mmread(f"{prefix}_pcmX.mtx")
        hz = scipy.io.mmread(f"{prefix}_pcmZ.mtx")
    else:
        from benchmark import load_code

        tanner = load_code(
            dict(label="C16", group="C16", A=[1, 4, 5, 11, 12, 15], B=[2, 6, 7, 9, 10, 14])
        )
        hx, hz = tanner.hx, tanner.hz
    return CssCode(label, hx, hz, rows_per_vertex)
