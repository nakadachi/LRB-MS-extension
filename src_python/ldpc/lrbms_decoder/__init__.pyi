import numpy as np
import scipy.sparse
from typing import Optional, List, Union
import ldpc.helpers.scipy_helpers
from ldpc.lrbms_decoder.check_groups import overlap_check_groups


class LrbmsDecoder:
    """
    Least-Reliable-Basis Min-Sum (LRB-MS) decoder with generalized checks.

    The rows of the parity-check matrix are grouped into generalized checks (GCs).
    Each GC of ``ell`` rows enforces its local syndrome constraint jointly, and
    returns min-sum (max-log) extrinsic LLRs computed over a candidate list built
    from its least-reliable basis. With one row per GC it reduces exactly to
    (normalised) min-sum BP.

    Parameters
    ----------
    pcm : Union[np.ndarray, scipy.sparse.spmatrix]
        The parity-check matrix.
    error_rate : Optional[float]
        Uniform bit/qubit error probability.
    error_channel : Optional[List[float]]
        Per-bit error probabilities (length ``n``). Overrides ``error_rate``.
    check_groups : Union[None, int, List[List[int]]]
        How rows are grouped into generalized checks.
        ``None`` -> one row per GC (plain min-sum);
        ``int ell`` -> greedy overlap grouping via
        :func:`ldpc.lrbms_decoder.overlap_check_groups`;
        list of lists -> explicit row groups (every non-zero row must be covered).
    max_iter : int
        Maximum number of message-passing iterations (0 -> ``n``).
    ms_scaling_factor : float
        Scaling factor applied to GC-to-variable messages.
    lrbms_order : int
        Order ``t`` of LRB-MS: besides the order-1 re-encodings of every
        most-reliable-basis position, all pairs among the ``t`` least-reliable
        MRB positions are added to the candidate list (``LRB-MS-t``).
    gc_method : str
        ``'lrbms'`` (default) or ``'trellis'`` (exact min-sum over the local coset,
        2^ell states; reference baseline, limited to ``ell <= max_trellis_ell``).
    schedule : str
        ``'parallel'`` (flooding) or ``'serial'`` (layered over GCs).
    llr_clip : float
        Magnitude clip for messages (also used for locally forced bits).
    max_trellis_ell : int
        Largest group size accepted by the trellis method.
    osd_method : str
        OSD post-processing applied when LRB-MS does not converge:
        ``'off'`` (default), ``'osd_0'``, ``'osd_cs'`` (combination sweep) or ``'osd_e'``.
        Uses the LRB-MS posterior LLRs as soft input, like ``BpOsdDecoder``.
    osd_order : int
        OSD order (ignored for ``'osd_0'``).
    """

    def __cinit__(self, pcm,
                  error_rate=None,
                  error_channel=None,
                  check_groups=None,
                  max_iter=0,
                  ms_scaling_factor=1.0,
                  lrbms_order=0,
                  gc_method="lrbms",
                  schedule="parallel",
                  llr_clip=50.0,
                  max_trellis_ell=22,
                  osd_method="off",
                  osd_order=0,
                  **kwargs): ...

    def __dealloc__(self): ...
    def _resolve_groups(self, H, check_groups): ...

    def _resolve_channel(self, error_rate, error_channel): ...
    def decode(self, syndrome) -> np.ndarray:
        """
        Decode a syndrome.

        Parameters
        ----------
        syndrome : np.ndarray
            Binary syndrome of length ``m``.

        Returns
        -------
        np.ndarray
            Estimated error of length ``n`` (hard decision of the final posteriors).
            Check :attr:`converge` to see whether it satisfies the syndrome.
        """
    @property
    def check_groups(self) -> List[List[int]]:
        """The row groups forming the generalized checks."""

    @check_groups.setter
    def check_groups(self, value) -> None: ...

    @property
    def max_group_size(self) -> int:
        """Largest number of rows (``ell``) in any generalized check."""

    @property
    def error_channel(self) -> np.ndarray: ...

    @error_channel.setter
    def error_channel(self, value) -> None: ...

    def update_channel_probs(self, value) -> None: ...

    @property
    def max_iter(self) -> int: ...

    @max_iter.setter
    def max_iter(self, value) -> None: ...

    @property
    def ms_scaling_factor(self) -> float: ...

    @ms_scaling_factor.setter
    def ms_scaling_factor(self, value) -> None: ...

    @property
    def lrbms_order(self) -> int: ...

    @lrbms_order.setter
    def lrbms_order(self, value) -> None: ...

    @property
    def gc_method(self) -> str: ...

    @gc_method.setter
    def gc_method(self, value) -> None: ...

    @property
    def schedule(self) -> str: ...

    @schedule.setter
    def schedule(self, value) -> None: ...

    @property
    def llr_clip(self) -> float: ...

    @llr_clip.setter
    def llr_clip(self, value) -> None: ...

    @property
    def decoding(self) -> np.ndarray:
        """Final output of the last decode (OSD result if the fallback was used)."""

    @property
    def log_prob_ratios(self) -> np.ndarray:
        """Posterior LLRs log(P(e_j=0)/P(e_j=1)) after the last decode."""

    @property
    def osd_method(self) -> str: ...

    @property
    def osd_order(self) -> int: ...

    @property
    def osd_used(self) -> bool:
        """True if the last decode fell back to OSD (LRB-MS did not converge)."""

    @property
    def lrbms_decoding(self) -> np.ndarray:
        """Hard decision of LRB-MS alone for the last decode (before any OSD fallback)."""

    @property
    def converge(self) -> bool: ...

    @property
    def iterations(self) -> int: ...

    @property
    def candidate_evaluations(self) -> int:
        """Cumulative number of scored candidates (trellis: state visits); a complexity counter."""

    def reset_counters(self) -> None: ...

    @property
    def pcm_shape(self): ...




def _rows_of(pcm):
    """Column indices of the nonzero entries of each row of a CSR matrix."""


class MbpLrbmsDecoder:
    """
    MBP4 + LRB-MS: quaternary memory-BP variable nodes with LRB-MS generalized checks.

    Decodes X and Z errors of a CSS code jointly (e.g. under depolarizing noise). Rows of
    ``hz`` are grouped into generalized checks acting on the x-part of the error, rows of
    ``hx`` into generalized checks acting on the z-part. Each check group is updated with the
    binary LRB-MS rule, its output scaled by ``mu``; each qubit keeps quaternary log-ratios
    updated with the MBP4 rule of Kuo and Lai: incoming check messages are weighted by
    ``1/alpha`` in the posterior, and the message back to a check subtracts that check's
    previous output without the ``1/alpha`` factor (memory / inhibition for ``alpha < 1``).

    Parameters
    ----------
    hx, hz : X-check and Z-check matrices (np.ndarray or scipy.sparse).
    error_rate : Optional[float]
        Depolarizing probability p (p_X = p_Y = p_Z = p/3).
    channel : Optional[tuple]
        ``(px, py, pz)``, each a float or a per-qubit sequence. Overrides ``error_rate``.
    x_groups, z_groups :
        Groupings of the rows of ``hx`` / ``hz`` (None, int ell, or list of row lists), as for
        :class:`LrbmsDecoder`.
    max_iter, mu, alpha, lrbms_order, schedule, llr_clip :
        Iterations; min-sum normalisation at the checks; MBP4 normalisation at the variable
        nodes; LRB-MS order; ``'serial'`` or ``'parallel'``; message clip.
    """

    def __cinit__(
        self,
        hx,
        hz,
        error_rate=None,
        channel=None,
        x_groups=None,
        z_groups=None,
        max_iter=100,
        mu=0.75,
        alpha=1.0,
        lrbms_order=0,
        schedule="serial",
        llr_clip=50.0,
    ): ...

    def __dealloc__(self): ...

    def _resolve_groups(self, pcm, groups): ...

    def _resolve_channel(self, error_rate, channel): ...

    def update_channel(self, error_rate=None, channel=None) -> None: ...

    def decode(self, syndrome_x, syndrome_z):
        """Decode; ``syndrome_x`` = hx @ e_z, ``syndrome_z`` = hz @ e_x. Returns ``(e_x, e_z)``."""

    @property
    def mu(self) -> float: ...

    @mu.setter
    def mu(self, value) -> None: ...

    @property
    def alpha(self) -> float: ...

    @alpha.setter
    def alpha(self, value) -> None: ...

    @property
    def max_iter(self) -> int: ...

    @max_iter.setter
    def max_iter(self, value) -> None: ...

    @property
    def schedule(self) -> str: ...

    @schedule.setter
    def schedule(self, value) -> None: ...

    @property
    def x_groups(self): ...

    @property
    def z_groups(self): ...

    @property
    def converge(self) -> bool: ...

    @property
    def iterations(self) -> int: ...
