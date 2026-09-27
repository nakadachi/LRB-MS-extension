#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np
ctypedef np.uint8_t uint8_t

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

cdef class LrbmsDecoder:
    cdef LrbmsDecoderCpp* lrbmsd
    cdef int m, n
    cdef vector[uint8_t] _syndrome
    cdef object _check_groups
    cdef bool MEMORY_ALLOCATED
