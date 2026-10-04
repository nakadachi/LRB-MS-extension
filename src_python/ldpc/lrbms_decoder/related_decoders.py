"""Decoders from related work on generalized-check decoding of quantum Tanner codes.

- :func:`union_overlap_check_groups`: the check-combining Algorithm 1 of Mostad, Rosnes and
  Lin, "Improved decoding of quantum Tanner codes using generalized check nodes",
  arXiv:2603.05486.
- :class:`GmbpDecoder`: their hybrid generalized MBP4 decoder: MBP4, then MBP4 with exact MAP
  generalized checks if MBP4 does not converge, then OSD.
- :class:`LeadDecoder`: LEAD of Xiao et al., "LEAD: a local ensemble-assisted parallel decoding
  framework for quantum Tanner codes", arXiv:2605.17796: BP-LSD on every local code, averaged
  into a prior for a global BP-OSD decoder.

SOGRAND (Rapp et al., arXiv:2603.18318) is the ``gc_method='sogrand'`` rule of
:class:`LrbmsDecoder` and :class:`MbpLrbmsDecoder`.
"""

from typing import List, Optional, Sequence, Union

import numpy as np
import scipy.sparse

from ldpc.bplsd_decoder import BpLsdDecoder
from ldpc.bposd_decoder import BpOsdDecoder
from ldpc.lrbms_decoder._lrbms_decoder import MbpLrbmsDecoder


def union_overlap_check_groups(
    pcm: Union[np.ndarray, scipy.sparse.spmatrix], r: int, seed: int = 0
) -> List[List[int]]:
    """Mostad et al., Algorithm 1: blocks of r and r - 1 rows grown from random seed rows.

    The block sizes are as many r's, then r - 1's, as needed to cover all rows. Each block
    starts from a uniformly random remaining row; it then repeatedly takes the remaining row
    with the largest overlap with the union of the block's supports (ties: lowest index).
    """
    if r < 1:
        raise ValueError("r must be >= 1")
    pcm = scipy.sparse.csr_matrix(pcm).astype(np.int64)
    row_count = pcm.shape[0]
    block_count = -(-row_count // r)
    small_blocks = block_count * r - row_count
    sizes = [r] * (block_count - small_blocks) + [r - 1] * small_blocks
    rng = np.random.default_rng(seed)
    remaining = list(range(row_count))
    groups = []
    for size in sizes:
        if not remaining:
            break
        first = remaining.pop(int(rng.integers(len(remaining))))
        block = [first]
        support = pcm[first].toarray().ravel() > 0
        for _ in range(size - 1):
            if not remaining:
                break
            overlaps = pcm[remaining] @ support.astype(np.int64)
            chosen = remaining.pop(int(np.argmax(overlaps)))
            block.append(chosen)
            support |= pcm[chosen].toarray().ravel() > 0
        groups.append(sorted(block))
    return groups


class GmbpDecoder:
    """Hybrid generalized MBP4 decoder of Mostad, Rosnes and Lin (arXiv:2603.05486).

    1. MBP4 (one check per row, box-plus checks) for ``first_stage_iter`` iterations.
    2. If it does not converge: MBP4 with the generalized checks ``x_groups`` / ``z_groups``,
       each decoded with the exact MAP rule, for ``max_iter`` iterations.
    3. If that does not converge either and ``osd_order`` is not None: OSD (combination sweep;
       order 1 is classical OSD-1) on each CSS half that still violates its syndrome.

    Both stages use their message rule with scaling ``a``: check outputs are scaled by 1/a in
    the posterior and in the messages (flooding schedule). The paper uses a = 1.6 (1.5 for the
    BB code), 6 iterations per stage and OSD-1; without OSD, 25 iterations in stage 2.

    Parameters
    ----------
    hx, hz : CSS check matrices.
    error_rate, channel : depolarizing rate p (p/3 per Pauli) or ``(px, py, pz)``.
    x_groups, z_groups : generalized checks of stage 2 (as for :class:`MbpLrbmsDecoder`).
    scaling : the scaling a.
    first_stage_iter : iterations of stage 1; 0 skips stage 1.
    max_iter : iterations of stage 2.
    osd_order : OSD order, or None for no OSD.
    gc_method : generalized-check rule of stage 2 (``'map'`` in the paper).
    **kwargs : passed to the stage-2 :class:`MbpLrbmsDecoder` (e.g. ``lrbms_order``).
    """

    def __init__(
        self,
        hx,
        hz,
        error_rate: Optional[float] = None,
        channel=None,
        x_groups=None,
        z_groups=None,
        scaling: float = 1.6,
        first_stage_iter: int = 6,
        max_iter: int = 6,
        osd_order: Optional[int] = 1,
        gc_method: str = "map",
        **kwargs,
    ):
        if scaling <= 0:
            raise ValueError("scaling must be positive.")
        common = dict(
            error_rate=error_rate, channel=channel, mu=1.0 / scaling, alpha=1.0, schedule="parallel"
        )
        self.first_stage = None
        if first_stage_iter > 0:
            self.first_stage = MbpLrbmsDecoder(
                hx, hz, max_iter=first_stage_iter, gc_method="map", **common
            )
        osd = {} if osd_order is None else dict(osd_method="osd_cs", osd_order=osd_order)
        self.second_stage = MbpLrbmsDecoder(
            hx,
            hz,
            x_groups=x_groups,
            z_groups=z_groups,
            max_iter=max_iter,
            gc_method=gc_method,
            **common,
            **osd,
            **kwargs,
        )
        self.stage = 0
        self.converge = False

    def update_channel(self, error_rate=None, channel=None) -> None:
        if self.first_stage is not None:
            self.first_stage.update_channel(error_rate=error_rate, channel=channel)
        self.second_stage.update_channel(error_rate=error_rate, channel=channel)

    def decode(self, syndrome_x, syndrome_z):
        """Returns ``(e_x, e_z)``; :attr:`stage` is 1, 2 or 3 (OSD) for the last decode."""
        if self.first_stage is not None:
            decoded = self.first_stage.decode(syndrome_x, syndrome_z)
            if self.first_stage.converge:
                self.stage, self.converge = 1, True
                return decoded
        decoded = self.second_stage.decode(syndrome_x, syndrome_z)
        self.converge = self.second_stage.converge
        self.stage = 3 if self.second_stage.osd_used else 2
        return decoded


class LeadDecoder:
    """LEAD (Xiao et al., arXiv:2605.17796) for one CSS half: decodes s = pcm @ e.

    1. Every local code (a group of rows, e.g. the checks of one Tanner-graph vertex) is
       decoded with BP-LSD from the channel prior and its local syndrome. Its soft output is
       p = 1 / (1 + exp(LLR)) of the BP stage; positions the local hard decision flips are
       raised to at least 0.5 when that decision satisfies the local syndrome.
    2. Each bit's prior is ``scaling`` times the mean of its local estimates.
    3. A global BP-OSD decoder runs from that prior.

    A local code with a zero syndrome is not decoded (ldpc's BP-LSD skips it as well); the paper
    does not say how such views enter the average. ``zero_views`` chooses: ``'prior'`` (the
    channel prior), ``'zero'`` (estimate 0: the view saw no error) or ``'skip'`` (left out of
    the average; bits seen by no other view keep the channel prior).

    Parameters
    ----------
    pcm : the check matrix of this CSS half.
    groups : the local codes, as lists of rows of ``pcm``.
    error_rate, error_channel : channel prior.
    scaling : the regularisation alpha in (0, 1] of the paper (1.0 by default).
    local_lsd_order, global_osd_order : LSD-CS and OSD-CS orders (3 in the paper).
    local_max_iter, global_max_iter : BP iterations; default the local / global code length.
    zero_views : treatment of local codes with a zero syndrome (see above).
    """

    def __init__(
        self,
        pcm,
        groups: Sequence[Sequence[int]],
        error_rate: Optional[float] = None,
        error_channel=None,
        scaling: float = 1.0,
        local_lsd_order: int = 3,
        global_osd_order: int = 3,
        local_max_iter: Optional[int] = None,
        global_max_iter: Optional[int] = None,
        zero_views: str = "prior",
    ):
        if zero_views not in ("prior", "zero", "skip"):
            raise ValueError("zero_views must be 'prior', 'zero' or 'skip'.")
        self.zero_views = zero_views
        self.pcm = scipy.sparse.csr_matrix(pcm).astype(np.uint8)
        bit_count = self.pcm.shape[1]
        if error_channel is not None:
            self.channel = np.asarray(error_channel, dtype=float)
        elif error_rate is not None:
            self.channel = np.full(bit_count, float(error_rate))
        else:
            raise ValueError("Please specify error_rate or error_channel.")
        self.scaling = float(scaling)
        self.locals = []
        for rows in groups:
            rows = sorted(int(row) for row in rows)
            local_rows = self.pcm[rows]
            support = np.unique(local_rows.indices)
            local_pcm = local_rows[:, support]
            decoder = BpLsdDecoder(
                local_pcm,
                error_channel=self.channel[support].tolist(),
                max_iter=local_max_iter or len(support),
                bp_method="minimum_sum",
                lsd_method="lsd_cs",
                lsd_order=local_lsd_order,
            )
            self.locals.append((rows, support, local_pcm, decoder))
        self.global_decoder = BpOsdDecoder(
            self.pcm,
            error_channel=self.channel.tolist(),
            max_iter=global_max_iter or bit_count,
            bp_method="minimum_sum",
            osd_method="osd_cs",
            osd_order=global_osd_order,
        )
        self.prior = self.channel.copy()

    def decode(self, syndrome) -> np.ndarray:
        syndrome = np.asarray(syndrome).astype(np.uint8) % 2
        totals = np.zeros(self.pcm.shape[1])
        counts = np.zeros(self.pcm.shape[1])
        for rows, support, local_pcm, decoder in self.locals:
            local_syndrome = syndrome[rows]
            if local_syndrome.any():
                local_error = decoder.decode(local_syndrome)
                estimate = 1.0 / (1.0 + np.exp(np.asarray(decoder.log_prob_ratios)))
                if np.array_equal(local_pcm @ local_error % 2, local_syndrome):
                    flipped = local_error.astype(bool)
                    estimate[flipped] = np.maximum(estimate[flipped], 0.5)
            elif self.zero_views == "skip":
                continue
            elif self.zero_views == "zero":
                estimate = np.zeros(len(support))
            else:
                estimate = self.channel[support]
            totals[support] += estimate
            counts[support] += 1
        prior = np.where(counts > 0, self.scaling * totals / np.maximum(counts, 1), self.channel)
        self.prior = np.clip(prior, 1e-12, 1 - 1e-12)
        self.global_decoder.update_channel_probs(self.prior)
        return self.global_decoder.decode(syndrome)
