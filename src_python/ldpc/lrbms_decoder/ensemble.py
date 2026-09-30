"""Ensemble LRB-MS: decode with several check groupings and keep the most likely correction."""

from typing import List, Optional, Sequence, Union

import numpy as np
import scipy.sparse

from ldpc.lrbms_decoder._lrbms_decoder import LrbmsDecoder
from ldpc.lrbms_decoder.check_groups import overlap_check_groups


def permuted_overlap_groupings(
    pcm: Union[np.ndarray, scipy.sparse.spmatrix], ell: int, count: int, seed: int = 0
) -> List[List[List[int]]]:
    """``count`` different greedy overlap groupings of the rows of ``pcm``.

    The first grouping is ``overlap_check_groups(pcm, ell)``; the others run the same
    greedy procedure on randomly permuted row orders, so the group boundaries differ.
    """
    pcm = scipy.sparse.csr_matrix(pcm)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        perm = np.arange(pcm.shape[0]) if i == 0 else rng.permutation(pcm.shape[0])
        groups = overlap_check_groups(pcm[perm], ell)
        out.append([sorted(int(perm[r]) for r in g) for g in groups])
    return out


class LrbmsEnsembleDecoder:
    """
    Runs :class:`LrbmsDecoder` once per check grouping and returns the
    syndrome-valid output of lowest channel cost ``sum_j e_j * log((1-p_j)/p_j)``.

    Different groupings put the generalized-check boundaries in different places,
    so their failures (non-convergence, or convergence to a heavier solution in the
    wrong coset) are only weakly correlated. Keeping the cheapest valid output
    never does worse, in channel cost, than any single member.

    Parameters
    ----------
    pcm : Union[np.ndarray, scipy.sparse.spmatrix]
        The parity-check matrix.
    error_rate, error_channel :
        As for :class:`LrbmsDecoder`.
    groupings : Optional[Sequence]
        Explicit list of groupings; each entry is anything
        :class:`LrbmsDecoder` accepts as ``check_groups``.
    ell, num_groupings, seed :
        Used when ``groupings`` is None: ``num_groupings`` greedy overlap
        groupings of size ``ell`` from :func:`permuted_overlap_groupings`.
    stop : str
        ``'all'`` (default) runs every member; ``'first'`` stops at the first
        member whose LRB-MS iterations converge (cheaper, slightly weaker).
    osd_members : str
        Which members get the OSD fallback given by ``osd_method``/``osd_order``:
        ``'all'`` (default) or ``'last'`` (only the final member, so OSD runs only
        once every other grouping has failed; cheapest with ``stop='first'``).
    **kwargs :
        Passed to every :class:`LrbmsDecoder` (``max_iter``, ``ms_scaling_factor``,
        ``lrbms_order``, ``schedule``, ``osd_method``, ``osd_order``, ...).
    """

    def __init__(self, pcm, error_rate: Optional[float] = None, error_channel=None,
                 groupings: Optional[Sequence] = None, ell: int = 8, num_groupings: int = 4,
                 seed: int = 0, stop: str = "all", osd_members: str = "all", **kwargs):
        if stop not in ("all", "first"):
            raise ValueError("stop must be 'all' or 'first'.")
        if osd_members not in ("all", "last"):
            raise ValueError("osd_members must be 'all' or 'last'.")
        self.pcm = scipy.sparse.csr_matrix(pcm).astype(np.uint8)
        n = self.pcm.shape[1]
        if groupings is None:
            if num_groupings < 1:
                raise ValueError("num_groupings must be >= 1.")
            groupings = permuted_overlap_groupings(self.pcm, ell, num_groupings, seed)
        if len(groupings) == 0:
            raise ValueError("At least one grouping is required.")
        if error_channel is not None:
            probs = np.asarray(error_channel, dtype=float)
        elif error_rate is not None:
            probs = np.full(n, float(error_rate))
        else:
            raise ValueError("Please specify error_rate or error_channel.")
        probs = np.clip(probs, 1e-300, 1 - 1e-16)
        self._weights = np.log((1 - probs) / probs)
        self.stop = stop
        no_osd = {k: v for k, v in kwargs.items() if k not in ("osd_method", "osd_order")}
        self.members = [
            LrbmsDecoder(self.pcm, error_channel=probs, check_groups=g,
                         **(kwargs if osd_members == "all" or i == len(groupings) - 1 else no_osd))
            for i, g in enumerate(groupings)
        ]
        self._converge = False
        self._member = -1
        self._decoding = np.zeros(n, dtype=np.uint8)

    def decode(self, syndrome) -> np.ndarray:
        """Decode a syndrome; returns the lowest-cost syndrome-valid member output.

        If no member produces a valid output, the first member's output is returned
        and :attr:`converge` is False.
        """
        syndrome = np.asarray(syndrome).astype(np.uint8) % 2
        best, best_cost, best_i, first = None, np.inf, -1, None
        for i, dec in enumerate(self.members):
            d = np.asarray(dec.decode(syndrome), dtype=np.uint8)
            if first is None:
                first = d
            if np.array_equal((self.pcm @ d) % 2, syndrome):
                cost = float(self._weights @ d)
                if cost < best_cost:
                    best, best_cost, best_i = d.copy(), cost, i
            if self.stop == "first" and dec.converge:
                break
        self._converge = best is not None
        self._member = best_i
        self._decoding = best if best is not None else first
        return self._decoding.copy()

    @property
    def converge(self) -> bool:
        """True if some member returned a syndrome-valid output for the last decode."""
        return self._converge

    @property
    def member(self) -> int:
        """Index of the member whose output was returned (-1 if none was valid)."""
        return self._member

    @property
    def decoding(self) -> np.ndarray:
        return self._decoding.copy()

    @property
    def groupings(self) -> List[List[List[int]]]:
        return [m.check_groups for m in self.members]
