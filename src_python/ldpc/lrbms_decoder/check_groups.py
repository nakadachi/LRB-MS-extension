"""Helpers for building generalized-check groupings for :class:`LrbmsDecoder`."""

from typing import List, Union

import numpy as np
import scipy.sparse


def _rows_as_sets(pcm: Union[np.ndarray, scipy.sparse.spmatrix]) -> List[set]:
    pcm = scipy.sparse.csr_matrix(pcm)
    pcm.eliminate_zeros()
    return [set(pcm.indices[pcm.indptr[i] : pcm.indptr[i + 1]].tolist()) for i in range(pcm.shape[0])]


def consecutive_check_groups(m: int, ell: int) -> List[List[int]]:
    """Partition rows ``0..m-1`` into consecutive blocks of ``ell`` rows.

    Useful when the parity-check matrix is already arranged so that neighbouring
    rows form a component code (e.g. GLDPC / product-like constructions).
    """
    if ell < 1:
        raise ValueError("ell must be >= 1")
    return [list(range(i, min(i + ell, m))) for i in range(0, m, ell)]


def overlap_check_groups(
    pcm: Union[np.ndarray, scipy.sparse.spmatrix], ell: int
) -> List[List[int]]:
    """Greedily partition the rows of ``pcm`` into groups of at most ``ell`` rows.

    Each group is seeded with the lowest-index unassigned row and grown by
    repeatedly adding the unassigned row sharing the most columns with the
    group's current support (ties broken by row index). Growth stops at ``ell``
    rows, or early when no remaining row overlaps the group (adding a disjoint
    row only enlarges the local code without coupling anything).
    """
    if ell < 1:
        raise ValueError("ell must be >= 1")
    rows = _rows_as_sets(pcm)
    m = len(rows)
    # column -> rows incidence for fast overlap scoring
    col_rows = {}
    for i, r in enumerate(rows):
        for j in r:
            col_rows.setdefault(j, []).append(i)
    unassigned = set(range(m))
    groups = []
    while unassigned:
        seed = min(unassigned)
        unassigned.discard(seed)
        group = [seed]
        support = set(rows[seed])
        while len(group) < ell:
            score = {}
            for j in support:
                for i in col_rows.get(j, ()):
                    if i in unassigned:
                        score[i] = score.get(i, 0) + 1
            if not score:
                break
            best = min(score, key=lambda i: (-score[i], i))
            group.append(best)
            unassigned.discard(best)
            support |= rows[best]
        groups.append(sorted(group))
    return groups
