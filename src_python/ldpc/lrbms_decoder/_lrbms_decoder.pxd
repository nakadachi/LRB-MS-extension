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
        MAP = 2
        SOGRAND = 3

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
        int sogrand_list_size
        double sogrand_threshold
        long long sogrand_max_queries
        int sogrand_intercept
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
        MbpLrbmsDecoderCpp(int qubit_count,
                           vector[vector[int]]& hx_rows,
                           vector[vector[int]]& hz_rows,
                           vector[vector[int]]& x_groups,
                           vector[vector[int]]& z_groups,
                           vector[double]& p_x,
                           vector[double]& p_y,
                           vector[double]& p_z,
                           int max_iter,
                           double mu,
                           double alpha,
                           int lrbms_order,
                           GcMethod gc_method,
                           LrbmsSchedule schedule,
                           double llr_clip) except +
        int maximum_iterations
        double alpha
        LrbmsSchedule schedule
        vector[uint8_t] error_x
        vector[uint8_t] error_z
        int iterations
        bool converge
        void set_channel(vector[double]& p_x, vector[double]& p_y, vector[double]& p_z) except +
        void set_mu(double mu)
        void set_gc_method(GcMethod gc_method)
        GcMethod gc_method()
        void set_sogrand(int list_size, double threshold, long long max_queries, int intercept)
        void marginal_llrs(vector[double]& llr_x, vector[double]& llr_z)
        void decode(vector[uint8_t]& syndrome_x, vector[uint8_t]& syndrome_z) except +

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
    cdef int qubit_count, x_check_count, z_check_count
    cdef double _mu
    cdef vector[uint8_t] _syndrome_x
    cdef vector[uint8_t] _syndrome_z
    cdef object _x_groups
    cdef object _z_groups
    cdef object _hx_csr
    cdef object _hz_csr
    cdef object _sogrand
    cdef bool MEMORY_ALLOCATED
    # OSD post-processing on each CSS half: x-part with H_Z, z-part with H_X
    cdef bool OSD_ALLOCATED
    cdef bool _osd_used
    cdef BpSparse* pcm_x_part
    cdef BpSparse* pcm_z_part
    cdef OsdDecoderCpp* osd_x_part
    cdef OsdDecoderCpp* osd_z_part
    cdef vector[double] _osd_channel_x
    cdef vector[double] _osd_channel_z
    cdef vector[double] _llr_x
    cdef vector[double] _llr_z
