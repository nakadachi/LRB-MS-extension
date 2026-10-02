#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np
ctypedef np.uint8_t uint8_t
from ldpc.bp_decoder._bp_decoder cimport BpSparse
from ldpc.bposd_decoder._bposd_decoder cimport OsdDecoderCpp, OsdMethod, OSD_OFF, OSD_0, EXHAUSTIVE, COMBINATION_SWEEP

cdef extern from "lrbms.hpp" namespace "ldpc::lrbms":

    cdef enum GcMethod:
        LRBMS = 0
        TRELLIS = 1

    cdef enum LrbmsSchedule:
        PARALLEL = 0
        SERIAL = 1

    cdef cppclass LrbmsDecoderCpp "ldpc::lrbms::LrbmsDecoder":
        LrbmsDecoderCpp(int m, int n,
                        vector[vector[int]]& rows,
                        vector[vector[int]]& check_groups,
                        vector[double]& channel_probs,
                        int max_iter,
                        double ms_scaling,
                        int order,
                        GcMethod method,
                        LrbmsSchedule sched,
                        double clip,
                        int max_trellis_ell) except +
        int check_count
        int bit_count
        int maximum_iterations
        double ms_scaling_factor
        int lrbms_order
        GcMethod gc_method
        LrbmsSchedule schedule
        double llr_clip
        int max_trellis_ell
        vector[uint8_t] decoding
        vector[double] log_prob_ratios
        vector[double] channel_probabilities
        int iterations
        bool converge
        long long candidate_evaluations
        vector[uint8_t]& decode(vector[uint8_t]& syndrome) except +
        void set_channel_probabilities(vector[double]& probs) except +
        void set_check_groups(vector[vector[int]]& groups) except +
        int max_group_size()

cdef extern from "mbp_lrbms.hpp" namespace "ldpc::lrbms":

    cdef cppclass MbpLrbmsDecoderCpp "ldpc::lrbms::MbpLrbmsDecoder":
        MbpLrbmsDecoderCpp(int n,
                           vector[vector[int]]& hx_rows,
                           vector[vector[int]]& hz_rows,
                           vector[vector[int]]& x_groups,
                           vector[vector[int]]& z_groups,
                           vector[double]& px,
                           vector[double]& py,
                           vector[double]& pz,
                           int max_iter,
                           double mu,
                           double alpha,
                           int order,
                           LrbmsSchedule sched,
                           double clip) except +
        int maximum_iterations
        double alpha
        LrbmsSchedule schedule
        vector[uint8_t] ex
        vector[uint8_t] ez
        int iterations
        bool converge
        void set_channel(vector[double]& px, vector[double]& py, vector[double]& pz) except +
        void set_mu(double mu)
        void decode(vector[uint8_t]& sx, vector[uint8_t]& sz) except +

cdef class LrbmsDecoder:
    cdef LrbmsDecoderCpp* lrbmsd
    cdef int m, n
    cdef vector[uint8_t] _syndrome
    cdef object _check_groups
    cdef bool MEMORY_ALLOCATED
    cdef BpSparse* pcm_bp
    cdef OsdDecoderCpp* osdD
    cdef vector[double] _osd_channel
    cdef vector[double] _llr
    cdef bool OSD_ALLOCATED
    cdef bool _osd_used

cdef class MbpLrbmsDecoder:
    cdef MbpLrbmsDecoderCpp* dec
    cdef int n, mx, mz
    cdef double _mu
    cdef vector[uint8_t] _sx
    cdef vector[uint8_t] _sz
    cdef object _x_groups
    cdef object _z_groups
    cdef bool MEMORY_ALLOCATED
