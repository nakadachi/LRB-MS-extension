"""Ensemble Qulid: decode with several check groupings and keep the most likely correction."""

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
    groupings = []
    for index in range(count):
        if index == 0:
            row_order = np.arange(pcm.shape[0])
        else:
            row_order = rng.permutation(pcm.shape[0])
        groups = overlap_check_groups(pcm[row_order], ell)
        groupings.append([sorted(int(row_order[row]) for row in group) for group in groups])
    return groupings


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
        ``'all'`` (default) runs every member. ``'escalate'`` accepts the first
        member's output when its Qulid iterations converge, and runs the remaining
        members only when they do not. In our benchmarks it matched ``'all'`` in
        accuracy at 2-3x less cost (use it with ``osd_members='all'``).
        ``'first'`` stops at the first member that converges (cheapest, weaker).
    osd_members : str
        Which members get the OSD fallback given by ``osd_method``/``osd_order``:
        ``'all'`` (default) or ``'last'`` (only the final member, so OSD runs only
        once every other grouping has failed; cheapest with ``stop='first'``).
    **kwargs :
        Passed to every :class:`LrbmsDecoder` (``max_iter``, ``ms_scaling_factor``,
        ``lrbms_order``, ``schedule``, ``osd_method``, ``osd_order``, ...).
    """

    def __init__(
        self,
        pcm,
        error_rate: Optional[float] = None,
        error_channel=None,
        groupings: Optional[Sequence] = None,
        ell: int = 8,
        num_groupings: int = 4,
        seed: int = 0,
        stop: str = "all",
        osd_members: str = "all",
        **kwargs,
    ):
        if stop not in ("all", "escalate", "first"):
            raise ValueError("stop must be 'all', 'escalate' or 'first'.")
        if osd_members not in ("all", "last"):
            raise ValueError("osd_members must be 'all' or 'last'.")
        self.pcm = scipy.sparse.csr_matrix(pcm).astype(np.uint8)
        bit_count = self.pcm.shape[1]
        if groupings is None:
            if num_groupings < 1:
                raise ValueError("num_groupings must be >= 1.")
            groupings = permuted_overlap_groupings(self.pcm, ell, num_groupings, seed)
        if len(groupings) == 0:
            raise ValueError("At least one grouping is required.")
        if error_channel is not None:
            probs = np.asarray(error_channel, dtype=float)
        elif error_rate is not None:
            probs = np.full(bit_count, float(error_rate))
        else:
            raise ValueError("Please specify error_rate or error_channel.")
        probs = np.clip(probs, 1e-300, 1 - 1e-16)
        self._weights = np.log((1 - probs) / probs)
        self.stop = stop
        kwargs_without_osd = {
            key: value for key, value in kwargs.items() if key not in ("osd_method", "osd_order")
        }
        last = len(groupings) - 1
        self.members = [
            LrbmsDecoder(
                self.pcm,
                error_channel=probs,
                check_groups=grouping,
                **(kwargs if osd_members == "all" or index == last else kwargs_without_osd),
            )
            for index, grouping in enumerate(groupings)
        ]
        self._converge = False
        self._member = -1
        self._decoding = np.zeros(bit_count, dtype=np.uint8)

    def decode(self, syndrome) -> np.ndarray:
        """Decode a syndrome; returns the lowest-cost syndrome-valid member output.

        If no member produces a valid output, the first member's output is returned
        and :attr:`converge` is False.
        """
        syndrome = np.asarray(syndrome).astype(np.uint8) % 2
        best_output, best_cost, best_index = None, np.inf, -1
        first_output = None
        for index, member in enumerate(self.members):
            output = np.asarray(member.decode(syndrome), dtype=np.uint8)
            if first_output is None:
                first_output = output
            if np.array_equal((self.pcm @ output) % 2, syndrome):
                cost = float(self._weights @ output)
                if cost < best_cost:
                    best_output, best_cost, best_index = output.copy(), cost, index
            stop_here = self.stop == "first" or (self.stop == "escalate" and index == 0)
            if member.converge and stop_here:
                break
        self._converge = best_output is not None
        self._member = best_index
        self._decoding = best_output if best_output is not None else first_output
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
        return [member.check_groups for member in self.members]
