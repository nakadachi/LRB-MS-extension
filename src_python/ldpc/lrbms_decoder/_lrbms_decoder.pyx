#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
import scipy.sparse
from typing import Optional, List, Union
import ldpc.helpers.scipy_helpers
from ldpc.lrbms_decoder.check_groups import overlap_check_groups

_GC_METHOD_NAMES = {LRBMS: "lrbms", TRELLIS: "trellis", MAP: "map", SOGRAND: "sogrand"}


cdef GcMethod _parse_gc_method(value) except *:
    name = str(value).lower()
    if name in ("qulid", "lrbms", "lrb_ms", "lrb-ms", "0"):
        return LRBMS
    if name in ("trellis", "exact", "maxlog", "max-log", "1"):
        return TRELLIS
    if name in ("map", "bcjr", "sum_product", "sum-product", "2"):
        return MAP
    if name in ("sogrand", "3"):
        return SOGRAND
    raise ValueError(
        f"gc_method '{value}' invalid. Choose 'qulid' (= 'lrbms'), 'trellis', 'map' or 'sogrand'."
    )


def _parse_osd_method(osd_method, osd_order):
    """(OsdMethod as int, order) for an OSD method name."""
    order = int(osd_order)
    method = str(osd_method).lower()
    if method in ("off", "osd_off", "none", "-1"):
        parsed = OSD_OFF
    elif method in ("osd_0", "osd0", "0"):
        parsed = OSD_0
        order = 0
    elif method in ("osd_cs", "cs", "combination_sweep", "1"):
        parsed = COMBINATION_SWEEP
    elif method in ("osd_e", "e", "exhaustive"):
        parsed = EXHAUSTIVE
    else:
        raise ValueError(
            f"osd_method '{osd_method}' invalid. Choose 'off', 'osd_0', 'osd_cs' or 'osd_e'."
        )
    if order < 0:
        raise ValueError("osd_order must be a non-negative integer.")
    return int(parsed), order


cdef BpSparse* _bp_sparse_of(pcm):
    """A new BpSparse copy of a CSR matrix (caller owns it)."""
    cdef BpSparse* matrix = new BpSparse(pcm.shape[0], pcm.shape[1], int(pcm.nnz))
    for row in range(pcm.shape[0]):
        for column in pcm.indices[pcm.indptr[row]:pcm.indptr[row + 1]]:
            matrix.insert_entry(row, int(column))
    return matrix


cdef class LrbmsDecoder:
    """
    Qulid (quantum least-reliable list decoding, formerly LRB-MS) decoder with generalized checks.

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
        Order ``t`` of Qulid: besides the order-1 re-encodings of every
        most-reliable-basis position, all pairs among the ``t`` least-reliable
        MRB positions are added to the candidate list (``Qulid-t``).
    gc_method : str
        ``'lrbms'`` (default; alias ``'qulid'``) or ``'trellis'`` (exact min-sum over the local coset,
        2^ell states; reference baseline, limited to ``ell <= max_trellis_ell``),
        ``'map'`` (exact sum-product over the same trellis, the generalized-check update of
        Mostad et al., arXiv:2603.05486) or ``'sogrand'`` (soft-output GRAND list decoding,
        Rapp et al., arXiv:2603.18318). Single-row groups use min-sum for ``'lrbms'`` and
        ``'trellis'`` and the box-plus (sum-product) rule for ``'map'`` and ``'sogrand'``.
    schedule : str
        ``'parallel'`` (flooding) or ``'serial'`` (layered over GCs).
    llr_clip : float
        Magnitude clip for messages (also used for locally forced bits).
    max_trellis_ell : int
        Largest group size accepted by the trellis method.
    osd_method : str
        OSD post-processing applied when Qulid does not converge:
        ``'off'`` (default), ``'osd_0'``, ``'osd_cs'`` (combination sweep) or ``'osd_e'``.
        Uses the Qulid posterior LLRs as soft input, like ``BpOsdDecoder``.
    osd_order : int
        OSD order (ignored for ``'osd_0'``).
    sogrand_list_size, sogrand_threshold, sogrand_max_queries, sogrand_intercept :
        SOGRAND settings: maximum list size; stop once the estimated probability that the
        true local pattern is not in the list is below the threshold; query limit per check
        update (0: none); ORBGRAND intercept (negative: chosen from the input LLRs).
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
                  sogrand_list_size=4,
                  sogrand_threshold=1e-5,
                  sogrand_max_queries=0,
                  sogrand_intercept=-1,
                  **kwargs):

        self.MEMORY_ALLOCATED = False
        self.OSD_ALLOCATED = False
        self._osd_used = False

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
        self.lrbmsd.sogrand_list_size = int(sogrand_list_size)
        self.lrbmsd.sogrand_threshold = float(sogrand_threshold)
        self.lrbmsd.sogrand_max_queries = int(sogrand_max_queries)
        self.lrbmsd.sogrand_intercept = int(sogrand_intercept)

        # optional OSD post-processing on Qulid failures (reuses ldpc's OSD implementation)
        parsed_method, parsed_order = _parse_osd_method(osd_method, osd_order)
        cdef OsdMethod om = <OsdMethod> parsed_method
        cdef int oo = parsed_order
        self._llr.resize(self.n)
        if om != OSD_OFF:
            self._osd_channel = cprobs
            self.pcm_bp = _bp_sparse_of(H)
            self.osdD = new OsdDecoderCpp(self.pcm_bp[0], om, oo, self._osd_channel)
            self.OSD_ALLOCATED = True

    def __dealloc__(self):
        if self.OSD_ALLOCATED:
            del self.osdD
            del self.pcm_bp
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
        self._osd_used = False
        self.lrbmsd.decode(self._syndrome)
        if self.lrbmsd.converge or not self.OSD_ALLOCATED:
            for i in range(self.n):
                out[i] = self.lrbmsd.decoding[i]
            return out
        for i in range(self.n):
            self._llr[i] = self.lrbmsd.log_prob_ratios[i]
        self.osdD.decode(self._syndrome, self._llr)
        self._osd_used = True
        for i in range(self.n):
            out[i] = self.osdD.osdw_decoding[i]
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
        cdef int i
        if self.OSD_ALLOCATED:
            for i in range(self.n):
                self._osd_channel[i] = cprobs[i]  # OSD holds a reference to this vector

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
        return _GC_METHOD_NAMES[self.lrbmsd.gc_method]

    @gc_method.setter
    def gc_method(self, value) -> None:
        self.lrbmsd.gc_method = _parse_gc_method(value)

    @property
    def sogrand_list_size(self) -> int:
        return self.lrbmsd.sogrand_list_size

    @sogrand_list_size.setter
    def sogrand_list_size(self, value) -> None:
        if value < 1:
            raise ValueError("sogrand_list_size must be at least 1.")
        self.lrbmsd.sogrand_list_size = int(value)

    @property
    def sogrand_threshold(self) -> float:
        return self.lrbmsd.sogrand_threshold

    @sogrand_threshold.setter
    def sogrand_threshold(self, value) -> None:
        self.lrbmsd.sogrand_threshold = float(value)

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
        """Final output of the last decode (OSD result if the fallback was used)."""
        out = np.zeros(self.n, dtype=np.uint8)
        for i in range(self.n):
            out[i] = self.osdD.osdw_decoding[i] if self._osd_used else self.lrbmsd.decoding[i]
        return out

    @property
    def log_prob_ratios(self) -> np.ndarray:
        """Posterior LLRs log(P(e_j=0)/P(e_j=1)) after the last decode."""
        out = np.zeros(self.n)
        for i in range(self.n):
            out[i] = self.lrbmsd.log_prob_ratios[i]
        return out

    @property
    def osd_method(self) -> str:
        if not self.OSD_ALLOCATED:
            return "OSD_OFF"
        if self.osdD.osd_method == OSD_0:
            return "OSD_0"
        if self.osdD.osd_method == COMBINATION_SWEEP:
            return "OSD_CS"
        if self.osdD.osd_method == EXHAUSTIVE:
            return "OSD_E"
        return "OSD_OFF"

    @property
    def osd_order(self) -> int:
        return self.osdD.osd_order if self.OSD_ALLOCATED else 0

    @property
    def osd_used(self) -> bool:
        """True if the last decode fell back to OSD (Qulid did not converge)."""
        return self._osd_used

    @property
    def lrbms_decoding(self) -> np.ndarray:
        """Hard decision of Qulid alone for the last decode (before any OSD fallback)."""
        out = np.zeros(self.n, dtype=np.uint8)
        for i in range(self.n):
            out[i] = self.lrbmsd.decoding[i]
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




def _rows_of(pcm):
    """Column indices of the nonzero entries of each row of a CSR matrix."""
    cdef vector[vector[int]] rows
    rows.resize(pcm.shape[0])
    for row in range(pcm.shape[0]):
        for column in pcm.indices[pcm.indptr[row]:pcm.indptr[row + 1]]:
            rows[row].push_back(int(column))
    return rows


cdef class MbpLrbmsDecoder:
    """
    MBP4 with generalized checks: quaternary memory-BP variable nodes, check groups updated
    with any binary generalized-check rule of :class:`LrbmsDecoder`.

    Decodes X and Z errors of a CSS code jointly (e.g. under depolarizing noise). Rows of
    ``hz`` are grouped into generalized checks acting on the x-part of the error, rows of
    ``hx`` into generalized checks acting on the z-part. Each check group is updated with the
    rule ``gc_method``, its output scaled by ``mu``; each qubit keeps quaternary log-ratios
    updated with the MBP4 rule of Kuo and Lai: incoming check messages are weighted by
    ``1/alpha`` in the posterior, and the message back to a check subtracts that check's
    previous output without the ``1/alpha`` factor (memory / inhibition for ``alpha < 1``).

    The generalized MBP4 decoder (GMBP4) of Mostad et al. (arXiv:2603.05486) with their
    scaling ``a`` is ``gc_method='map', mu=1/a, alpha=1, schedule='parallel'``.

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
        Iterations; normalisation at the checks; MBP4 normalisation at the variable nodes;
        Qulid order; ``'serial'`` or ``'parallel'``; message clip.
    gc_method : str
        ``'lrbms'`` (default), ``'trellis'``, ``'map'`` or ``'sogrand'``, as for
        :class:`LrbmsDecoder`.
    osd_method, osd_order :
        OSD post-processing when the decoder does not converge, as for :class:`LrbmsDecoder`.
        It runs on each CSS half whose hard decision does not match its syndrome, with the
        binary marginals of the quaternary posteriors as soft input. Classical OSD-1 (all
        weight-1 flips) is ``osd_method='osd_cs', osd_order=1``.
    sogrand_list_size, sogrand_threshold, sogrand_max_queries, sogrand_intercept :
        SOGRAND settings, as for :class:`LrbmsDecoder`.
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
        gc_method="lrbms",
        osd_method="off",
        osd_order=0,
        sogrand_list_size=4,
        sogrand_threshold=1e-5,
        sogrand_max_queries=0,
        sogrand_intercept=-1,
    ):
        self.MEMORY_ALLOCATED = False
        self.OSD_ALLOCATED = False
        self._osd_used = False
        hx_csr = scipy.sparse.csr_matrix(ldpc.helpers.scipy_helpers.convert_to_binary_sparse(hx))
        hz_csr = scipy.sparse.csr_matrix(ldpc.helpers.scipy_helpers.convert_to_binary_sparse(hz))
        hx_csr.eliminate_zeros()
        hz_csr.eliminate_zeros()
        if hx_csr.shape[1] != hz_csr.shape[1]:
            raise ValueError("hx and hz must have the same number of columns.")
        if ((hx_csr @ hz_csr.T).toarray() % 2).any():
            raise ValueError("hx and hz do not commute (not a CSS code).")
        self.qubit_count = hx_csr.shape[1]
        self.x_check_count = hx_csr.shape[0]
        self.z_check_count = hz_csr.shape[0]
        self._hx_csr = hx_csr
        self._hz_csr = hz_csr
        self._x_groups = self._resolve_groups(hx_csr, x_groups)
        self._z_groups = self._resolve_groups(hz_csr, z_groups)
        p_x, p_y, p_z = self._resolve_channel(error_rate, channel)
        schedule_name = str(schedule).lower()
        if schedule_name not in ("serial", "parallel"):
            raise ValueError("schedule must be 'serial' or 'parallel'.")
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        cdef GcMethod method = _parse_gc_method(gc_method)
        parsed_osd_method, parsed_osd_order = _parse_osd_method(osd_method, osd_order)

        cdef vector[vector[int]] hx_rows = _rows_of(hx_csr)
        cdef vector[vector[int]] hz_rows = _rows_of(hz_csr)
        cdef vector[vector[int]] x_group_rows = self._x_groups
        cdef vector[vector[int]] z_group_rows = self._z_groups
        cdef vector[double] p_x_vec = p_x
        cdef vector[double] p_y_vec = p_y
        cdef vector[double] p_z_vec = p_z
        self.dec = new MbpLrbmsDecoderCpp(
            self.qubit_count,
            hx_rows,
            hz_rows,
            x_group_rows,
            z_group_rows,
            p_x_vec,
            p_y_vec,
            p_z_vec,
            int(max_iter),
            float(mu),
            float(alpha),
            int(lrbms_order),
            method,
            SERIAL if schedule_name == "serial" else PARALLEL,
            float(llr_clip),
        )
        self.MEMORY_ALLOCATED = True
        self._mu = float(mu)
        self._syndrome_x.resize(self.x_check_count)
        self._syndrome_z.resize(self.z_check_count)
        self.sogrand_settings = (sogrand_list_size, sogrand_threshold, sogrand_max_queries,
                                 sogrand_intercept)

        cdef OsdMethod om = <OsdMethod> parsed_osd_method
        if om != OSD_OFF:
            self._set_osd_channels(p_x, p_y, p_z)
            self.pcm_x_part = _bp_sparse_of(hz_csr)
            self.pcm_z_part = _bp_sparse_of(hx_csr)
            self.osd_x_part = new OsdDecoderCpp(self.pcm_x_part[0], om, parsed_osd_order,
                                                self._osd_channel_x)
            self.osd_z_part = new OsdDecoderCpp(self.pcm_z_part[0], om, parsed_osd_order,
                                                self._osd_channel_z)
            self.OSD_ALLOCATED = True

    def __dealloc__(self):
        if self.OSD_ALLOCATED:
            del self.osd_x_part
            del self.osd_z_part
            del self.pcm_x_part
            del self.pcm_z_part
        if self.MEMORY_ALLOCATED:
            del self.dec

    def _set_osd_channels(self, p_x, p_y, p_z):
        """Marginal flip probabilities of each CSS half (in place: OSD keeps a reference)."""
        self._osd_channel_x.resize(self.qubit_count)
        self._osd_channel_z.resize(self.qubit_count)
        for i in range(self.qubit_count):
            self._osd_channel_x[i] = p_x[i] + p_y[i]
            self._osd_channel_z[i] = p_z[i] + p_y[i]

    def _resolve_groups(self, pcm, groups):
        if groups is None:
            return [[row] for row in range(pcm.shape[0])]
        if isinstance(groups, (int, np.integer)):
            return overlap_check_groups(pcm, int(groups))
        resolved = [[int(row) for row in group] for group in groups]
        for group in resolved:
            for row in group:
                if row < 0 or row >= pcm.shape[0]:
                    raise ValueError(
                        f"Check group row index {row} out of range [0, {pcm.shape[0]})."
                    )
        return resolved

    def _resolve_channel(self, error_rate, channel):
        if channel is not None:
            if len(channel) != 3:
                raise ValueError("channel must be (px, py, pz).")
            return [
                list(np.broadcast_to(np.asarray(probs, dtype=float), (self.qubit_count,)))
                for probs in channel
            ]
        if error_rate is None:
            raise ValueError("Please specify error_rate (depolarizing) or channel=(px, py, pz).")
        return [[float(error_rate) / 3.0] * self.qubit_count for _ in range(3)]

    def update_channel(self, error_rate=None, channel=None) -> None:
        p_x, p_y, p_z = self._resolve_channel(error_rate, channel)
        cdef vector[double] p_x_vec = p_x
        cdef vector[double] p_y_vec = p_y
        cdef vector[double] p_z_vec = p_z
        self.dec.set_channel(p_x_vec, p_y_vec, p_z_vec)
        if self.OSD_ALLOCATED:
            self._set_osd_channels(p_x, p_y, p_z)

    def decode(self, syndrome_x, syndrome_z):
        """Decode; ``syndrome_x`` = hx @ e_z, ``syndrome_z`` = hz @ e_x. Returns ``(e_x, e_z)``."""
        cdef int i
        cdef uint8_t bit
        syndrome_x = np.asarray(syndrome_x)
        syndrome_z = np.asarray(syndrome_z)
        if syndrome_x.shape[0] != self.x_check_count or syndrome_z.shape[0] != self.z_check_count:
            raise ValueError("Syndrome lengths do not match hx / hz.")
        # Assign through a typed local: Cython cannot write a conditional expression
        # directly into a C++ vector element.
        for i in range(self.x_check_count):
            bit = 1 if syndrome_x[i] else 0
            self._syndrome_x[i] = bit
        for i in range(self.z_check_count):
            bit = 1 if syndrome_z[i] else 0
            self._syndrome_z[i] = bit
        self.dec.decode(self._syndrome_x, self._syndrome_z)
        error_x = np.empty(self.qubit_count, dtype=np.uint8)
        error_z = np.empty(self.qubit_count, dtype=np.uint8)
        for i in range(self.qubit_count):
            error_x[i] = self.dec.error_x[i]
            error_z[i] = self.dec.error_z[i]
        self._osd_used = False
        if self.dec.converge or not self.OSD_ALLOCATED:
            return error_x, error_z

        self.dec.marginal_llrs(self._llr_x, self._llr_z)
        if not np.array_equal((self._hz_csr @ error_x) % 2, syndrome_z != 0):
            self.osd_x_part.decode(self._syndrome_z, self._llr_x)
            for i in range(self.qubit_count):
                error_x[i] = self.osd_x_part.osdw_decoding[i]
            self._osd_used = True
        if not np.array_equal((self._hx_csr @ error_z) % 2, syndrome_x != 0):
            self.osd_z_part.decode(self._syndrome_x, self._llr_z)
            for i in range(self.qubit_count):
                error_z[i] = self.osd_z_part.osdw_decoding[i]
            self._osd_used = True
        return error_x, error_z

    @property
    def mu(self) -> float:
        return self._mu

    @mu.setter
    def mu(self, value) -> None:
        self._mu = float(value)
        self.dec.set_mu(self._mu)

    @property
    def alpha(self) -> float:
        return self.dec.alpha

    @alpha.setter
    def alpha(self, value) -> None:
        if value <= 0:
            raise ValueError("alpha must be positive.")
        self.dec.alpha = float(value)

    @property
    def max_iter(self) -> int:
        return self.dec.maximum_iterations

    @max_iter.setter
    def max_iter(self, value) -> None:
        self.dec.maximum_iterations = int(value)

    @property
    def schedule(self) -> str:
        return "serial" if self.dec.schedule == SERIAL else "parallel"

    @schedule.setter
    def schedule(self, value) -> None:
        schedule_name = str(value).lower()
        if schedule_name not in ("serial", "parallel"):
            raise ValueError("schedule must be 'serial' or 'parallel'.")
        self.dec.schedule = SERIAL if schedule_name == "serial" else PARALLEL

    @property
    def x_groups(self):
        return self._x_groups

    @property
    def z_groups(self):
        return self._z_groups

    @property
    def converge(self) -> bool:
        return self.dec.converge

    @property
    def iterations(self) -> int:
        return self.dec.iterations

    @property
    def gc_method(self) -> str:
        return _GC_METHOD_NAMES[self.dec.gc_method()]

    @gc_method.setter
    def gc_method(self, value) -> None:
        self.dec.set_gc_method(_parse_gc_method(value))

    @property
    def osd_used(self) -> bool:
        """True if the last decode fell back to OSD on at least one CSS half."""
        return self._osd_used

    @property
    def sogrand_settings(self):
        """(list_size, threshold, max_queries, intercept) of the SOGRAND rule."""
        return self._sogrand

    @sogrand_settings.setter
    def sogrand_settings(self, value) -> None:
        list_size, threshold, max_queries, intercept = value
        if list_size < 1:
            raise ValueError("The SOGRAND list size must be at least 1.")
        self._sogrand = (int(list_size), float(threshold), int(max_queries), int(intercept))
        self.dec.set_sogrand(self._sogrand[0], self._sogrand[1], self._sogrand[2],
                             self._sogrand[3])
