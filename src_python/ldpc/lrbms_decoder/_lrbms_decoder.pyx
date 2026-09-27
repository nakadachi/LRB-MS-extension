#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
import scipy.sparse
from typing import Optional, List, Union
import ldpc.helpers.scipy_helpers
from ldpc.lrbms_decoder.check_groups import overlap_check_groups


cdef class LrbmsDecoder:
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
                  **kwargs):

        self.MEMORY_ALLOCATED = False

        for key in kwargs.keys():
            if key not in ["channel_probs"]:
                raise ValueError(f"Unknown parameter '{key}' passed to the LrbmsDecoder constructor.")
        if error_channel is None and "channel_probs" in kwargs:
            error_channel = kwargs["channel_probs"]

        if not (isinstance(pcm, np.ndarray) or isinstance(pcm, scipy.sparse.spmatrix)
                or scipy.sparse.issparse(pcm)):
            raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray "
                            f"or scipy.sparse.spmatrix object, not {type(pcm)}")

        H = scipy.sparse.csr_matrix(ldpc.helpers.scipy_helpers.convert_to_binary_sparse(pcm))
        H.eliminate_zeros()
        self.m, self.n = H.shape

        cdef vector[vector[int]] rows
        rows.resize(self.m)
        cdef int i
        for i in range(self.m):
            for j in H.indices[H.indptr[i]:H.indptr[i + 1]]:
                rows[i].push_back(int(j))

        groups = self._resolve_groups(H, check_groups)
        cdef vector[vector[int]] cgroups = groups

        probs = self._resolve_channel(error_rate, error_channel)
        cdef vector[double] cprobs = probs

        if max_iter < 0:
            raise ValueError("max_iter must be a non-negative integer.")
        if lrbms_order < 0:
            raise ValueError("lrbms_order must be a non-negative integer.")

        self.lrbmsd = new LrbmsDecoderCpp(self.m, self.n, rows, cgroups, cprobs,
                                          int(max_iter), float(ms_scaling_factor), int(lrbms_order),
                                          LRBMS, PARALLEL, float(llr_clip), int(max_trellis_ell))
        self.MEMORY_ALLOCATED = True
        self._syndrome.resize(self.m)
        self._check_groups = groups
        self.gc_method = gc_method
        self.schedule = schedule

    def __dealloc__(self):
        if self.MEMORY_ALLOCATED:
            del self.lrbmsd

    # ------------------------------------------------------------------ helpers
    def _resolve_groups(self, H, check_groups):
        if check_groups is None:
            return [[i] for i in range(self.m)]
        if isinstance(check_groups, (int, np.integer)):
            return overlap_check_groups(H, int(check_groups))
        groups = [[int(r) for r in g] for g in check_groups]
        for g in groups:
            for r in g:
                if r < 0 or r >= self.m:
                    raise ValueError(f"Check group row index {r} out of range [0, {self.m}).")
        return groups

    def _resolve_channel(self, error_rate, error_channel):
        if error_channel is not None:
            probs = [float(p) for p in error_channel]
            if len(probs) != self.n:
                raise ValueError(f"The error channel must have length {self.n}, not {len(probs)}.")
        elif error_rate is not None:
            probs = [float(error_rate)] * self.n
        else:
            raise ValueError("Please specify the error channel. Either: 1) error_rate: float or "
                             "2) error_channel: list of floats of length equal to the block length of the code.")
        return probs

    # ------------------------------------------------------------------ decode
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
        cdef int i
        cdef uint8_t bit
        syndrome = np.asarray(syndrome)
        if syndrome.shape[0] != self.m:
            raise ValueError(f"The syndrome must have length {self.m}. Not {syndrome.shape[0]}.")
        zero = True
        for i in range(self.m):
            bit = 1 if syndrome[i] else 0
            self._syndrome[i] = bit
            if bit:
                zero = False
        out = np.zeros(self.n, dtype=syndrome.dtype)
        if zero:
            self.lrbmsd.converge = True
            self.lrbmsd.iterations = 0
            for i in range(self.n):
                self.lrbmsd.decoding[i] = 0
            return out
        self.lrbmsd.decode(self._syndrome)
        for i in range(self.n):
            out[i] = self.lrbmsd.decoding[i]
        return out

    # --------------------------------------------------------------- properties
    @property
    def check_groups(self) -> List[List[int]]:
        """The row groups forming the generalized checks."""
        return [list(g) for g in self._check_groups]

    @check_groups.setter
    def check_groups(self, value) -> None:
        H = None
        if isinstance(value, (int, np.integer)):
            raise ValueError("Set an integer group size at construction time, or pass explicit groups "
                             "(e.g. from ldpc.lrbms_decoder.overlap_check_groups).")
        groups = self._resolve_groups(H, value)
        cdef vector[vector[int]] cgroups = groups
        self.lrbmsd.set_check_groups(cgroups)
        self._check_groups = groups

    @property
    def max_group_size(self) -> int:
        """Largest number of rows (``ell``) in any generalized check."""
        return self.lrbmsd.max_group_size()

    @property
    def error_channel(self) -> np.ndarray:
        out = np.zeros(self.n)
        for i in range(self.n):
            out[i] = self.lrbmsd.channel_probabilities[i]
        return out

    @error_channel.setter
    def error_channel(self, value) -> None:
        probs = self._resolve_channel(None, value)
        cdef vector[double] cprobs = probs
        self.lrbmsd.set_channel_probabilities(cprobs)

    def update_channel_probs(self, value) -> None:
        self.error_channel = value

    @property
    def max_iter(self) -> int:
        return self.lrbmsd.maximum_iterations

    @max_iter.setter
    def max_iter(self, value) -> None:
        if value < 0:
            raise ValueError("max_iter must be a non-negative integer.")
        self.lrbmsd.maximum_iterations = value

    @property
    def ms_scaling_factor(self) -> float:
        return self.lrbmsd.ms_scaling_factor

    @ms_scaling_factor.setter
    def ms_scaling_factor(self, value) -> None:
        self.lrbmsd.ms_scaling_factor = value

    @property
    def lrbms_order(self) -> int:
        return self.lrbmsd.lrbms_order

    @lrbms_order.setter
    def lrbms_order(self, value) -> None:
        if value < 0:
            raise ValueError("lrbms_order must be a non-negative integer.")
        self.lrbmsd.lrbms_order = value

    @property
    def gc_method(self) -> str:
        return "trellis" if self.lrbmsd.gc_method == TRELLIS else "lrbms"

    @gc_method.setter
    def gc_method(self, value) -> None:
        v = str(value).lower()
        if v in ("lrbms", "lrb_ms", "lrb-ms", "0"):
            self.lrbmsd.gc_method = LRBMS
        elif v in ("trellis", "exact", "map", "1"):
            self.lrbmsd.gc_method = TRELLIS
        else:
            raise ValueError(f"gc_method '{value}' invalid. Choose 'lrbms' or 'trellis'.")

    @property
    def schedule(self) -> str:
        return "serial" if self.lrbmsd.schedule == SERIAL else "parallel"

    @schedule.setter
    def schedule(self, value) -> None:
        v = str(value).lower()
        if v in ("parallel", "flooding", "0"):
            self.lrbmsd.schedule = PARALLEL
        elif v in ("serial", "layered", "1"):
            self.lrbmsd.schedule = SERIAL
        else:
            raise ValueError(f"schedule '{value}' invalid. Choose 'parallel' or 'serial'.")

    @property
    def llr_clip(self) -> float:
        return self.lrbmsd.llr_clip

    @llr_clip.setter
    def llr_clip(self, value) -> None:
        self.lrbmsd.llr_clip = value

    @property
    def decoding(self) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.uint8)
        for i in range(self.n):
            out[i] = self.lrbmsd.decoding[i]
        return out

    @property
    def log_prob_ratios(self) -> np.ndarray:
        """Posterior LLRs log(P(e_j=0)/P(e_j=1)) after the last decode."""
        out = np.zeros(self.n)
        for i in range(self.n):
            out[i] = self.lrbmsd.log_prob_ratios[i]
        return out

    @property
    def converge(self) -> bool:
        return self.lrbmsd.converge

    @property
    def iterations(self) -> int:
        return self.lrbmsd.iterations

    @property
    def candidate_evaluations(self) -> int:
        """Cumulative number of scored candidates (trellis: state visits); a complexity counter."""
        return self.lrbmsd.candidate_evaluations

    def reset_counters(self) -> None:
        self.lrbmsd.candidate_evaluations = 0

    @property
    def pcm_shape(self):
        return (self.m, self.n)

