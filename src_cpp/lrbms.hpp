#ifndef LDPC_LRBMS_HPP
#define LDPC_LRBMS_HPP

/**
 * Qulid (quantum least-reliable list decoding, formerly LRB-MS: Least-Reliable-Basis Min-Sum)
 * decoding with generalized checks.
 *
 * The rows of the parity-check matrix H are partitioned (or covered) by
 * "generalized checks" (GCs). A GC c groups ell_c rows of H; its local
 * constraint is H_c e_{S_c} = s_c, where S_c is the union of the supports
 * of its rows and s_c the corresponding syndrome bits. Messages are passed
 * between variable nodes and GCs; each GC computes max-log (min-sum)
 * extrinsic LLRs over its local syndrome coset.
 *
 * GC update rules:
 *  - LRBMS   : reliability-ordered Gaussian elimination of H_c picks the
 *              least-reliable basis (LRB, the pivot columns). The most reliable
 *              positions are fixed to their hard decisions, the LRB is solved
 *              from the syndrome (candidate e0). The candidate list consists of
 *              e0, every order-1 re-encoding (flip one non-pivot position, re-solve
 *              the LRB) and, for order t > 0, all pairs among the t least-reliable
 *              non-pivot positions. Extrinsic LLRs are min-sum differences over the
 *              list. With a single parity check per GC this is exactly min-sum.
 *  - TRELLIS : exact max-log (min-sum) over the full local coset via a syndrome
 *              trellis with 2^ell states. Reference / "exact MAP-MS" baseline.
 *  - MAP     : exact sum-product (BCJR) over the same syndrome trellis: the
 *              generalized-check update of Mostad, Rosnes and Lin (arXiv:2603.05486).
 *              With one row per GC it is the box-plus rule, i.e. sum-product BP.
 *  - SOGRAND : soft-output GRAND (Rapp et al., arXiv:2603.18318; Yuan et al. 2025).
 *              Error patterns are queried in ORBGRAND order around the hard
 *              decision until the list holds `sogrand_list_size` syndrome-consistent
 *              patterns or the estimated probability that the true pattern is not
 *              in the list drops below `sogrand_threshold`. That probability is
 *              2^-rank times the probability mass not yet queried; it is spread over
 *              the bits according to their input marginals. Output is extrinsic.
 * MAP and SOGRAND use the box-plus rule for single-row GCs.
 */

#include <vector>
#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

namespace ldpc {
namespace lrbms {

enum GcMethod {
    LRBMS = 0,
    TRELLIS = 1,
    MAP = 2,
    SOGRAND = 3
};

enum LrbmsSchedule {
    PARALLEL = 0,
    SERIAL = 1
};

struct GeneralizedCheck {
    std::vector<int> rows;      // global row indices
    std::vector<int> support;   // global column indices (sorted, unique)
    int ell = 0;                // number of rows
    int rank = 0;               // GF(2) rank of the local rows
    int nwords = 0;             // 64-bit words per local row bitset
    std::vector<uint64_t> row_bits;  // ell * nwords, local row bitsets
    std::vector<uint64_t> col_masks; // per local column: ell-bit mask (TRELLIS only, ell <= 63)
    int msg_offset = 0;         // offset into flat message arrays
};

class LrbmsDecoder {
public:
    int check_count;
    int bit_count;
    std::vector<std::vector<int>> pcm_rows;
    std::vector<GeneralizedCheck> gcs;
    std::vector<double> channel_probabilities;
    std::vector<double> channel_llrs;
    int maximum_iterations;
    double ms_scaling_factor;
    int lrbms_order;
    GcMethod gc_method;
    LrbmsSchedule schedule;
    double llr_clip;
    int max_trellis_ell;

    // SOGRAND settings (defaults follow Rapp et al.)
    int sogrand_list_size = 4;
    double sogrand_threshold = 1e-5;     // stop once P(true pattern not in list) < threshold
    long long sogrand_max_queries = 0;   // 0: no limit
    int sogrand_intercept = -1;          // ORBGRAND intercept; < 0: chosen from the input LLRs

    std::vector<uint8_t> decoding;
    std::vector<double> log_prob_ratios;
    std::vector<uint8_t> candidate_syndrome;
    int iterations = 0;
    bool converge = false;
    long long candidate_evaluations = 0; // complexity counter (candidates scored)

    // messages, flat, indexed by gc.msg_offset + local position
    std::vector<double> c2v;
    std::vector<double> c2v_new;
    std::vector<int> msg_var;   // global variable index for each flat message slot
    int total_msgs = 0;

    LrbmsDecoder(int m, int n,
                 const std::vector<std::vector<int>> &rows,
                 const std::vector<std::vector<int>> &check_groups,
                 const std::vector<double> &channel_probs,
                 int max_iter = 0,
                 double ms_scaling = 1.0,
                 int order = 0,
                 GcMethod method = LRBMS,
                 LrbmsSchedule sched = PARALLEL,
                 double clip = 50.0,
                 int max_trellis_ell_ = 22)
        : check_count(m), bit_count(n), pcm_rows(rows),
          maximum_iterations(max_iter), ms_scaling_factor(ms_scaling), lrbms_order(order),
          gc_method(method), schedule(sched), llr_clip(clip), max_trellis_ell(max_trellis_ell_) {
        if ((int) rows.size() != m) throw std::invalid_argument("LrbmsDecoder: rows.size() != m");
        for (auto &r: pcm_rows) {
            std::sort(r.begin(), r.end());
            r.erase(std::unique(r.begin(), r.end()), r.end());
            for (int j: r) if (j < 0 || j >= n) throw std::invalid_argument("LrbmsDecoder: column index out of range");
        }
        this->set_check_groups(check_groups);
        this->set_channel_probabilities(channel_probs);
        decoding.assign(n, 0);
        log_prob_ratios.assign(n, 0.0);
        candidate_syndrome.assign(m, 0);
    }

    void set_channel_probabilities(const std::vector<double> &probs) {
        if ((int) probs.size() != bit_count) throw std::invalid_argument("LrbmsDecoder: channel probability vector has wrong length");
        channel_probabilities = probs;
        channel_llrs.resize(bit_count);
        for (int j = 0; j < bit_count; j++) {
            double p = probs[j];
            if (p <= 0.0) channel_llrs[j] = llr_clip;
            else if (p >= 1.0) channel_llrs[j] = -llr_clip;
            else channel_llrs[j] = std::log((1.0 - p) / p);
        }
    }

    void set_check_groups(const std::vector<std::vector<int>> &groups_in) {
        std::vector<std::vector<int>> groups = groups_in;
        if (groups.empty()) {
            groups.resize(check_count);
            for (int i = 0; i < check_count; i++) groups[i] = {i};
        }
        std::vector<uint8_t> covered(check_count, 0);
        gcs.clear();
        total_msgs = 0;
        msg_var.clear();
        std::vector<int> local_index(bit_count, -1);
        for (auto &g: groups) {
            if (g.empty()) continue;
            GeneralizedCheck gc;
            gc.rows = g;
            std::sort(gc.rows.begin(), gc.rows.end());
            gc.rows.erase(std::unique(gc.rows.begin(), gc.rows.end()), gc.rows.end());
            for (int r: gc.rows) {
                if (r < 0 || r >= check_count) throw std::invalid_argument("LrbmsDecoder: check group row index out of range");
                covered[r] = 1;
                for (int j: pcm_rows[r]) gc.support.push_back(j);
            }
            std::sort(gc.support.begin(), gc.support.end());
            gc.support.erase(std::unique(gc.support.begin(), gc.support.end()), gc.support.end());
            if (gc.support.empty()) continue; // all-zero rows carry no information
            gc.ell = (int) gc.rows.size();
            int s = (int) gc.support.size();
            gc.nwords = (s + 63) / 64;
            for (int k = 0; k < s; k++) local_index[gc.support[k]] = k;
            gc.row_bits.assign((size_t) gc.ell * gc.nwords, 0ULL);
            for (int i = 0; i < gc.ell; i++) {
                for (int j: pcm_rows[gc.rows[i]]) {
                    int k = local_index[j];
                    gc.row_bits[(size_t) i * gc.nwords + (k >> 6)] |= (1ULL << (k & 63));
                }
            }
            {   // GF(2) rank of the group's rows
                std::vector<uint64_t> T(gc.row_bits);
                int rk = 0;
                for (int k = 0; k < s && rk < gc.ell; k++) {
                    int w = k >> 6; uint64_t bit = 1ULL << (k & 63);
                    int piv = -1;
                    for (int i = rk; i < gc.ell; i++) if (T[(size_t) i * gc.nwords + w] & bit) { piv = i; break; }
                    if (piv < 0) continue;
                    for (int q = 0; q < gc.nwords; q++) std::swap(T[(size_t) piv * gc.nwords + q], T[(size_t) rk * gc.nwords + q]);
                    for (int i = 0; i < gc.ell; i++)
                        if (i != rk && (T[(size_t) i * gc.nwords + w] & bit))
                            for (int q = 0; q < gc.nwords; q++) T[(size_t) i * gc.nwords + q] ^= T[(size_t) rk * gc.nwords + q];
                    rk++;
                }
                gc.rank = rk;
            }
            if (gc.ell <= 63) {
                gc.col_masks.assign(s, 0ULL);
                for (int i = 0; i < gc.ell; i++)
                    for (int j: pcm_rows[gc.rows[i]]) gc.col_masks[local_index[j]] |= (1ULL << i);
            }
            for (int k = 0; k < s; k++) local_index[gc.support[k]] = -1;
            gc.msg_offset = total_msgs;
            for (int j: gc.support) msg_var.push_back(j);
            total_msgs += s;
            gcs.push_back(std::move(gc));
        }
        for (int i = 0; i < check_count; i++) {
            if (!covered[i] && !pcm_rows[i].empty())
                throw std::invalid_argument("LrbmsDecoder: row " + std::to_string(i) + " is not covered by any check group");
        }
        c2v.assign(total_msgs, 0.0);
        c2v_new.assign(total_msgs, 0.0);
    }

    int max_group_size() const {
        int mx = 0;
        for (auto &gc: gcs) mx = std::max(mx, gc.ell);
        return mx;
    }

    // ------------------------------------------------------------------ decode
    std::vector<uint8_t> &decode(const std::vector<uint8_t> &syndrome) {
        if ((int) syndrome.size() != check_count) throw std::invalid_argument("LrbmsDecoder: syndrome has wrong length");
        if ((gc_method == TRELLIS || gc_method == MAP) && max_group_size() > max_trellis_ell)
            throw std::invalid_argument("LrbmsDecoder: TRELLIS and MAP methods limited to groups with ell <= " + std::to_string(max_trellis_ell));
        if (gc_method == SOGRAND && max_group_size() > 63)
            throw std::invalid_argument("LrbmsDecoder: SOGRAND method limited to groups with ell <= 63");

        int max_iter = maximum_iterations > 0 ? maximum_iterations : bit_count;
        converge = false;
        iterations = 0;
        std::fill(c2v.begin(), c2v.end(), 0.0);

        // local syndromes
        std::vector<std::vector<uint8_t>> local_synd(gcs.size());
        for (size_t c = 0; c < gcs.size(); c++) {
            local_synd[c].resize(gcs[c].ell);
            for (int i = 0; i < gcs[c].ell; i++) local_synd[c][i] = syndrome[gcs[c].rows[i]] & 1;
        }

        // posterior = channel + sum c2v
        log_prob_ratios = channel_llrs;

        std::vector<double> in, out;
        for (int it = 1; it <= max_iter; it++) {
            iterations = it;
            if (schedule == PARALLEL) {
                for (size_t c = 0; c < gcs.size(); c++) {
                    auto &gc = gcs[c];
                    int s = (int) gc.support.size();
                    in.resize(s);
                    out.resize(s);
                    for (int k = 0; k < s; k++)
                        in[k] = log_prob_ratios[gc.support[k]] - c2v[gc.msg_offset + k];
                    gc_update(gc, local_synd[c], in, out);
                    for (int k = 0; k < s; k++) c2v_new[gc.msg_offset + k] = out[k];
                }
                c2v.swap(c2v_new);
                log_prob_ratios = channel_llrs;
                for (int t = 0; t < total_msgs; t++) log_prob_ratios[msg_var[t]] += c2v[t];
            } else { // SERIAL (layered over generalized checks)
                for (size_t c = 0; c < gcs.size(); c++) {
                    auto &gc = gcs[c];
                    int s = (int) gc.support.size();
                    in.resize(s);
                    out.resize(s);
                    for (int k = 0; k < s; k++)
                        in[k] = log_prob_ratios[gc.support[k]] - c2v[gc.msg_offset + k];
                    gc_update(gc, local_synd[c], in, out);
                    for (int k = 0; k < s; k++) {
                        c2v[gc.msg_offset + k] = out[k];
                        log_prob_ratios[gc.support[k]] = in[k] + out[k];
                    }
                }
            }

            for (int j = 0; j < bit_count; j++) decoding[j] = log_prob_ratios[j] < 0.0 ? 1 : 0;
            if (syndrome_matches(syndrome)) {
                converge = true;
                break;
            }
        }
        return decoding;
    }

    bool syndrome_matches(const std::vector<uint8_t> &syndrome) {
        bool ok = true;
        for (int i = 0; i < check_count; i++) {
            uint8_t b = 0;
            for (int j: pcm_rows[i]) b ^= decoding[j];
            candidate_syndrome[i] = b;
            if (b != (syndrome[i] & 1)) ok = false;
        }
        return ok;
    }

    // --------------------------------------------------------- GC updates
    void gc_update(const GeneralizedCheck &gc, const std::vector<uint8_t> &synd,
                   const std::vector<double> &in, std::vector<double> &out) {
        const bool soft_rule = gc_method == MAP || gc_method == SOGRAND;
        if (gc.ell == 1 && soft_rule) single_check_sum_product(gc, synd, in, out);
        else if (gc.ell == 1) single_check_update(gc, synd, in, out);
        else if (gc_method == TRELLIS) trellis_update(gc, synd, in, out);
        else if (gc_method == MAP) map_update(gc, synd, in, out);
        else if (gc_method == SOGRAND) sogrand_update(gc, synd, in, out);
        else lrbms_update(gc, synd, in, out);
        for (auto &x: out) {
            x *= ms_scaling_factor;
            if (x > llr_clip) x = llr_clip;
            else if (x < -llr_clip) x = -llr_clip;
        }
    }

    // scratch buffers (reused across calls)
    std::vector<int> order_buf, pivot_of_row, nonpivots;
    std::vector<uint8_t> is_pivot, hard, e0, cur;
    std::vector<double> absl, best0, best1;
    std::vector<uint64_t> R, hmask;
    std::vector<uint8_t> sb;
    std::vector<int> diff;

    void lrbms_update(const GeneralizedCheck &gc, const std::vector<uint8_t> &synd,
                      const std::vector<double> &lam, std::vector<double> &out) {
        const int s = (int) gc.support.size();
        const int W = gc.nwords;
        const int L = gc.ell;
        const double INF = std::numeric_limits<double>::infinity();

        absl.resize(s);
        hard.resize(s);
        for (int k = 0; k < s; k++) {
            absl[k] = std::fabs(lam[k]);
            hard[k] = lam[k] < 0.0 ? 1 : 0;
        }
        // min-heap on (|lambda|, index): positions are popped in increasing reliability only as
        // far as needed (rank of H_c for the elimination, plus t for the order-2 list)
        auto less_rel = [&](int a, int b) { return absl[a] < absl[b] || (absl[a] == absl[b] && a < b); };
        auto heap_cmp = [&](int a, int b) { return less_rel(b, a); };
        order_buf.resize(s);
        std::iota(order_buf.begin(), order_buf.end(), 0);
        std::make_heap(order_buf.begin(), order_buf.end(), heap_cmp);
        auto pop_least = [&]() {
            std::pop_heap(order_buf.begin(), order_buf.end(), heap_cmp);
            int k = order_buf.back();
            order_buf.pop_back();
            return k;
        };
        nonpivots.clear();

        // reliability-ordered Gaussian elimination -> least reliable basis
        R.assign(gc.row_bits.begin(), gc.row_bits.end());
        sb.assign(synd.begin(), synd.end());
        is_pivot.assign(s, 0);
        pivot_of_row.assign(L, -1);
        int r = 0;
        while (r < gc.rank && !order_buf.empty()) {
            int k = pop_least();
            int w = k >> 6;
            uint64_t bit = 1ULL << (k & 63);
            int piv = -1;
            for (int i = r; i < L; i++) if (R[(size_t) i * W + w] & bit) { piv = i; break; }
            if (piv < 0) { nonpivots.push_back(k); continue; }
            if (piv != r) {
                for (int q = 0; q < W; q++) std::swap(R[(size_t) piv * W + q], R[(size_t) r * W + q]);
                std::swap(sb[piv], sb[r]);
            }
            for (int i = 0; i < L; i++) {
                if (i != r && (R[(size_t) i * W + w] & bit)) {
                    for (int q = 0; q < W; q++) R[(size_t) i * W + q] ^= R[(size_t) r * W + q];
                    sb[i] ^= sb[r];
                }
            }
            pivot_of_row[r] = k;
            is_pivot[k] = 1;
            r++;
        }
        for (int i = r; i < L; i++) {
            if (sb[i]) { // inconsistent local syndrome: no information
                std::fill(out.begin(), out.end(), 0.0);
                return;
            }
        }
        // non-pivot positions: the t least reliable in order, the rest in any order
        {
            int need = lrbms_order;
            while ((int) nonpivots.size() < need && !order_buf.empty()) nonpivots.push_back(pop_least());
            for (int k: order_buf) nonpivots.push_back(k);
        }

        // e0: MRB = hard decisions, LRB solved from syndrome
        hmask.assign(W, 0ULL);
        for (int k = 0; k < s; k++) if (hard[k] && !is_pivot[k]) hmask[k >> 6] |= (1ULL << (k & 63));
        e0.assign(hard.begin(), hard.end());
        for (int i = 0; i < r; i++) {
            int par = sb[i];
            for (int q = 0; q < W; q++) par ^= __builtin_parityll(R[(size_t) i * W + q] & hmask[q]);
            e0[pivot_of_row[i]] = (uint8_t) par;
        }
        double cost0 = 0.0;
        for (int i = 0; i < r; i++) {
            int p = pivot_of_row[i];
            if (e0[p] != hard[p]) cost0 += absl[p];
        }

        if (L <= 64) {
            lrbms_score_fast(s, W, r, cost0, out);
            return;
        }

        // ---- build candidate list as difference sets relative to e0 ----
        // candidate 0 is e0 itself (empty diff)
        cand_cost.clear();
        cand_ptr.clear();
        cand_idx.clear();
        cand_ptr.push_back(0);
        cand_cost.push_back(cost0);
        cand_ptr.push_back(0);

        auto push_candidate = [&]() {
            double cost = cost0;
            for (int k: diff) {
                cost += (e0[k] != hard[k]) ? -absl[k] : absl[k];
                cand_idx.push_back(k);
            }
            cand_cost.push_back(cost);
            cand_ptr.push_back((int) cand_idx.size());
        };


        // order-1 re-encodings
        for (int j: nonpivots) {
            diff.clear();
            diff.push_back(j);
            uint64_t bit = 1ULL << (j & 63);
            int w = j >> 6;
            for (int i = 0; i < r; i++) if (R[(size_t) i * W + w] & bit) diff.push_back(pivot_of_row[i]);
            push_candidate();
        }
        // order-2 re-encodings among the t least reliable MRB positions
        int t = std::min<int>(lrbms_order, (int) nonpivots.size());
        for (int a = 0; a < t; a++) {
            for (int b = a + 1; b < t; b++) {
                int j1 = nonpivots[a], j2 = nonpivots[b];
                diff.clear();
                diff.push_back(j1);
                diff.push_back(j2);
                uint64_t b1 = 1ULL << (j1 & 63), b2 = 1ULL << (j2 & 63);
                int w1 = j1 >> 6, w2 = j2 >> 6;
                for (int i = 0; i < r; i++) {
                    bool x = ((R[(size_t) i * W + w1] & b1) != 0) ^ ((R[(size_t) i * W + w2] & b2) != 0);
                    if (x) diff.push_back(pivot_of_row[i]);
                }
                push_candidate();
            }
        }
        const int nc = (int) cand_cost.size();
        candidate_evaluations += nc;

        // ---- extrinsic minima in O(sum |diff| + s) ----
        // flip_min[k]: best cost among candidates that flip k relative to e0
        // keep_min[k]: best cost among candidates that keep e0[k]
        flip_min.assign(s, INF);
        keep_min.assign(s, INF);
        for (int c = 1; c < nc; c++) {
            double cc = cand_cost[c];
            for (int q = cand_ptr[c]; q < cand_ptr[c + 1]; q++) {
                int k = cand_idx[q];
                if (cc < flip_min[k]) flip_min[k] = cc;
            }
        }
        // keep_min: scan candidates in increasing cost; each resolves the positions it does not flip
        cand_order.resize(nc);
        std::iota(cand_order.begin(), cand_order.end(), 0);
        std::sort(cand_order.begin(), cand_order.end(), [&](int x, int y) { return cand_cost[x] < cand_cost[y]; });
        unresolved.resize(s);
        std::iota(unresolved.begin(), unresolved.end(), 0);
        stamp.assign(s, -1);
        for (int oc = 0; oc < nc && !unresolved.empty(); oc++) {
            int c = cand_order[oc];
            for (int q = cand_ptr[c]; q < cand_ptr[c + 1]; q++) stamp[cand_idx[q]] = c;
            int keep = 0;
            for (int u = 0; u < (int) unresolved.size(); u++) {
                int k = unresolved[u];
                if (stamp[k] == c) unresolved[keep++] = k;   // flipped by c: stays unresolved
                else keep_min[k] = cand_cost[c];
            }
            unresolved.resize(keep);
        }

        for (int k = 0; k < s; k++) {
            // own contribution removed: [value != hard] * |lambda_k|
            uint8_t v_keep = e0[k], v_flip = e0[k] ^ 1;
            double ek = keep_min[k] < INF ? keep_min[k] - (v_keep != hard[k] ? absl[k] : 0.0) : INF;
            double ef = flip_min[k] < INF ? flip_min[k] - (v_flip != hard[k] ? absl[k] : 0.0) : INF;
            double m0 = v_keep == 0 ? ek : ef;
            double m1 = v_keep == 0 ? ef : ek;
            bool f0 = m0 < INF, f1 = m1 < INF;
            if (f0 && f1) out[k] = m1 - m0;
            else if (f0) out[k] = llr_clip;   // bit forced to 0 by the local syndrome
            else if (f1) out[k] = -llr_clip;  // bit forced to 1
            else out[k] = 0.0;
        }
    }

    // Candidate scoring for ell <= 64: each candidate is (base positions, pivot-row mask).
    // Costs, flip minima and keep minima are computed without materialising candidate vectors.
    std::vector<uint64_t> colrow;          // per local column: mask of pivot rows containing it
    std::vector<double> delta, pdelta;     // cost change for flipping position k / pivot of row i
    std::vector<uint64_t> cand_rm;         // candidate pivot-row masks
    std::vector<int> cand_j1, cand_j2;     // candidate base positions (-1 = none)

    void lrbms_score_fast(int s, int W, int r, double cost0, std::vector<double> &out) {
        const double INF = std::numeric_limits<double>::infinity();
        delta.resize(s);
        for (int k = 0; k < s; k++) delta[k] = (e0[k] != hard[k]) ? -absl[k] : absl[k];
        pdelta.resize(r);
        for (int i = 0; i < r; i++) pdelta[i] = delta[pivot_of_row[i]];

        // column -> pivot-row incidence of the reduced matrix (non-pivot columns only)
        colrow.assign(s, 0ULL);
        for (int i = 0; i < r; i++) {
            const uint64_t rb = 1ULL << i;
            for (int q = 0; q < W; q++) {
                uint64_t word = R[(size_t) i * W + q];
                while (word) {
                    int k = (q << 6) + __builtin_ctzll(word);
                    word &= word - 1;
                    colrow[k] |= rb;
                }
            }
        }
        for (int i = 0; i < r; i++) colrow[pivot_of_row[i]] = 0ULL;

        auto rm_cost = [&](uint64_t rm) {
            double c = 0.0;
            while (rm) { c += pdelta[__builtin_ctzll(rm)]; rm &= rm - 1; }
            return c;
        };

        cand_cost.clear(); cand_rm.clear(); cand_j1.clear(); cand_j2.clear();
        cand_cost.push_back(cost0); cand_rm.push_back(0ULL); cand_j1.push_back(-1); cand_j2.push_back(-1);

        for (int j: nonpivots) {
            uint64_t rm = colrow[j];
            cand_cost.push_back(cost0 + delta[j] + rm_cost(rm));
            cand_rm.push_back(rm); cand_j1.push_back(j); cand_j2.push_back(-1);
        }
        int t = std::min<int>(lrbms_order, (int) nonpivots.size());
        for (int a = 0; a < t; a++) {
            for (int b = a + 1; b < t; b++) {
                int j1 = nonpivots[a], j2 = nonpivots[b];
                uint64_t rm = colrow[j1] ^ colrow[j2];
                cand_cost.push_back(cost0 + delta[j1] + delta[j2] + rm_cost(rm));
                cand_rm.push_back(rm); cand_j1.push_back(j1); cand_j2.push_back(j2);
            }
        }
        const int nc = (int) cand_cost.size();
        candidate_evaluations += nc;

        // flip minima
        flip_min.assign(s, INF);
        int cbest = 0;
        for (int c = 0; c < nc; c++) {
            double cc = cand_cost[c];
            if (cc < cand_cost[cbest]) cbest = c;
            if (c == 0) continue;
            if (cc < flip_min[cand_j1[c]]) flip_min[cand_j1[c]] = cc;
            if (cand_j2[c] >= 0 && cc < flip_min[cand_j2[c]]) flip_min[cand_j2[c]] = cc;
            uint64_t rm = cand_rm[c];
            while (rm) {
                int k = pivot_of_row[__builtin_ctzll(rm)];
                rm &= rm - 1;
                if (cc < flip_min[k]) flip_min[k] = cc;
            }
        }
        // keep minima: the global best keeps every position it does not flip
        const double dbest = cand_cost[cbest];
        keep_min.assign(s, dbest);
        auto keep_scan_base = [&](int j) {
            double m = INF;
            for (int c = 0; c < nc; c++)
                if (cand_j1[c] != j && cand_j2[c] != j && cand_cost[c] < m) m = cand_cost[c];
            keep_min[j] = m;
        };
        if (cand_j1[cbest] >= 0) keep_scan_base(cand_j1[cbest]);
        if (cand_j2[cbest] >= 0) keep_scan_base(cand_j2[cbest]);
        uint64_t rmb = cand_rm[cbest];
        while (rmb) {
            int i = __builtin_ctzll(rmb);
            rmb &= rmb - 1;
            const uint64_t bit = 1ULL << i;
            double m = INF;
            for (int c = 0; c < nc; c++)
                if (!(cand_rm[c] & bit) && cand_cost[c] < m) m = cand_cost[c];
            keep_min[pivot_of_row[i]] = m;
        }

        for (int k = 0; k < s; k++) {
            uint8_t v_keep = e0[k], v_flip = e0[k] ^ 1;
            double ek = keep_min[k] < INF ? keep_min[k] - (v_keep != hard[k] ? absl[k] : 0.0) : INF;
            double ef = flip_min[k] < INF ? flip_min[k] - (v_flip != hard[k] ? absl[k] : 0.0) : INF;
            double m0 = v_keep == 0 ? ek : ef;
            double m1 = v_keep == 0 ? ef : ek;
            bool f0 = m0 < INF, f1 = m1 < INF;
            if (f0 && f1) out[k] = m1 - m0;
            else if (f0) out[k] = llr_clip;
            else if (f1) out[k] = -llr_clip;
            else out[k] = 0.0;
        }
    }

    // exact min-sum for a single parity check (ell == 1)
    void single_check_update(const GeneralizedCheck &gc, const std::vector<uint8_t> &synd,
                             const std::vector<double> &lam, std::vector<double> &out) {
        const int s = (int) gc.support.size();
        double min1 = std::numeric_limits<double>::infinity(), min2 = min1;
        int idx = -1;
        int par = synd[0] & 1;
        for (int k = 0; k < s; k++) {
            double a = std::fabs(lam[k]);
            if (lam[k] < 0.0) par ^= 1;
            if (a < min1) { min2 = min1; min1 = a; idx = k; }
            else if (a < min2) min2 = a;
        }
        candidate_evaluations += s;
        for (int k = 0; k < s; k++) {
            int pk = par ^ (lam[k] < 0.0 ? 1 : 0);   // parity of the other hard decisions (+ syndrome)
            double mag = (k == idx) ? min2 : min1;
            if (s == 1) mag = llr_clip;               // weight-1 check: bit fully determined
            out[k] = pk ? -mag : mag;
        }
    }

    std::vector<double> cand_cost, flip_min, keep_min;
    std::vector<int> cand_ptr, cand_idx, cand_order, unresolved, stamp;

    std::vector<double> F, Bcur, Bnext;

    void trellis_update(const GeneralizedCheck &gc, const std::vector<uint8_t> &synd,
                        const std::vector<double> &lam, std::vector<double> &out) {
        const int s = (int) gc.support.size();
        const int L = gc.ell;
        const size_t S = (size_t) 1 << L;
        const double INF = std::numeric_limits<double>::infinity();
        uint64_t target = 0;
        for (int i = 0; i < L; i++) if (synd[i]) target |= (1ULL << i);

        // forward: F[k][state] = min cost of positions 0..k-1 reaching state
        F.assign((size_t) (s + 1) * S, INF);
        F[0] = 0.0;
        for (int k = 0; k < s; k++) {
            const double *Fk = &F[(size_t) k * S];
            double *Fn = &F[(size_t) (k + 1) * S];
            uint64_t cm = gc.col_masks[k];
            double c1 = lam[k];
            for (size_t st = 0; st < S; st++) {
                double v = Fk[st];
                if (v == INF) continue;
                if (v < Fn[st]) Fn[st] = v;
                size_t st1 = st ^ cm;
                if (v + c1 < Fn[st1]) Fn[st1] = v + c1;
            }
        }
        // backward: B[state] = min cost of positions k..s-1 driving state to target
        Bnext.assign(S, INF);
        Bnext[target] = 0.0;
        Bcur.resize(S);
        for (int k = s - 1; k >= 0; k--) {
            const double *Fk = &F[(size_t) k * S];
            uint64_t cm = gc.col_masks[k];
            double m0 = INF, m1 = INF;
            for (size_t st = 0; st < S; st++) {
                double f = Fk[st];
                if (f == INF) continue;
                double a0 = f + Bnext[st];
                double a1 = f + Bnext[st ^ cm];
                if (a0 < m0) m0 = a0;
                if (a1 < m1) m1 = a1;
            }
            candidate_evaluations += (long long) S;
            if (m0 < INF && m1 < INF) out[k] = m1 - m0;
            else if (m0 < INF) out[k] = llr_clip;
            else if (m1 < INF) out[k] = -llr_clip;
            else out[k] = 0.0;
            double c1 = lam[k];
            for (size_t st = 0; st < S; st++) {
                double a0 = Bnext[st];
                double a1 = Bnext[st ^ cm] + c1;
                Bcur[st] = a0 < a1 ? a0 : a1;
            }
            Bcur.swap(Bnext);
        }
    }

    // ------------------------------------------------------- sum-product rules
    // 2 atanh(tanh(a/2) tanh(b/2)), stable for large |a|, |b|; +inf is the neutral element
    static double box_plus(double a, double b) {
        const double sign = ((a < 0.0) != (b < 0.0)) ? -1.0 : 1.0;
        const double magnitude = std::min(std::fabs(a), std::fabs(b));
        return sign * magnitude + std::log1p(std::exp(-std::fabs(a + b))) -
               std::log1p(std::exp(-std::fabs(a - b)));
    }

    std::vector<double> prefix_bp, suffix_bp;

    // exact sum-product for a single parity check (ell == 1)
    void single_check_sum_product(const GeneralizedCheck &gc, const std::vector<uint8_t> &synd,
                                  const std::vector<double> &lam, std::vector<double> &out) {
        const int s = (int) gc.support.size();
        const double sign = (synd[0] & 1) ? -1.0 : 1.0;
        candidate_evaluations += s;
        if (s == 1) {
            out[0] = sign * llr_clip;   // weight-1 check: bit fully determined
            return;
        }
        const double INF = std::numeric_limits<double>::infinity();
        prefix_bp.resize(s + 1);
        suffix_bp.resize(s + 1);
        prefix_bp[0] = INF;
        for (int k = 0; k < s; k++) prefix_bp[k + 1] = box_plus(prefix_bp[k], lam[k]);
        suffix_bp[s] = INF;
        for (int k = s - 1; k >= 0; k--) suffix_bp[k] = box_plus(suffix_bp[k + 1], lam[k]);
        for (int k = 0; k < s; k++) out[k] = sign * box_plus(prefix_bp[k], suffix_bp[k + 1]);
    }

    std::vector<double> prob0, prob1, Fmap, Bmap, Bmap_next;

    // Exact MAP (sum-product) over the local coset with a 2^ell-state syndrome trellis.
    // Forward and backward vectors keep unit total mass (p0 + p1 = 1 at every step), so no
    // rescaling is needed; on underflow the GC falls back to the max-log trellis.
    void map_update(const GeneralizedCheck &gc, const std::vector<uint8_t> &synd,
                    const std::vector<double> &lam, std::vector<double> &out) {
        const int s = (int) gc.support.size();
        const int L = gc.ell;
        const size_t S = (size_t) 1 << L;
        uint64_t target = 0;
        for (int i = 0; i < L; i++) if (synd[i]) target |= (1ULL << i);

        prob0.resize(s);
        prob1.resize(s);
        for (int k = 0; k < s; k++) {
            const double e = std::exp(-std::fabs(lam[k]));
            const double likely = 1.0 / (1.0 + e), unlikely = e / (1.0 + e);
            prob0[k] = lam[k] >= 0.0 ? likely : unlikely;
            prob1[k] = lam[k] >= 0.0 ? unlikely : likely;
        }

        // forward: Fmap[k][state] = P(bits 0..k-1 produce partial syndrome `state`)
        Fmap.assign((size_t) (s + 1) * S, 0.0);
        Fmap[0] = 1.0;
        for (int k = 0; k < s; k++) {
            const double *Fk = &Fmap[(size_t) k * S];
            double *Fn = &Fmap[(size_t) (k + 1) * S];
            const uint64_t cm = gc.col_masks[k];
            const double p0 = prob0[k], p1 = prob1[k];
            for (size_t st = 0; st < S; st++) {
                const double v = Fk[st];
                if (v == 0.0) continue;
                Fn[st] += v * p0;
                Fn[st ^ cm] += v * p1;
            }
        }

        // backward: Bmap_next[state] = P(bits k+1..s-1 move `state` to the target syndrome)
        Bmap_next.assign(S, 0.0);
        Bmap_next[target] = 1.0;
        Bmap.resize(S);
        for (int k = s - 1; k >= 0; k--) {
            const double *Fk = &Fmap[(size_t) k * S];
            const uint64_t cm = gc.col_masks[k];
            double P0 = 0.0, P1 = 0.0;   // coset probability with bit k = 0 / 1, own prior excluded
            for (size_t st = 0; st < S; st++) {
                const double f = Fk[st];
                if (f == 0.0) continue;
                P0 += f * Bmap_next[st];
                P1 += f * Bmap_next[st ^ cm];
            }
            candidate_evaluations += (long long) S;
            if (P0 > 0.0 && P1 > 0.0) out[k] = std::log(P0) - std::log(P1);
            else if (P0 > 0.0) out[k] = llr_clip;
            else if (P1 > 0.0) out[k] = -llr_clip;
            else {
                trellis_update(gc, synd, lam, out);   // underflow or inconsistent syndrome
                return;
            }
            const double p0 = prob0[k], p1 = prob1[k];
            for (size_t st = 0; st < S; st++) Bmap[st] = Bmap_next[st] * p0 + Bmap_next[st ^ cm] * p1;
            Bmap.swap(Bmap_next);
        }
    }

    // ------------------------------------------------------------------ SOGRAND
    std::vector<int> orb_position;        // local position of each reliability rank (rank 1 = index 0)
    std::vector<double> orb_reliability;  // |lambda| in rank order
    std::vector<uint64_t> orb_mask;       // column syndrome mask in rank order
    std::vector<int> orb_choice;          // ranks of the pattern being built
    std::vector<int> sog_list_ranks, sog_list_ptr;
    std::vector<double> sog_list_prob, sog_p0, sog_p1;
    std::vector<uint8_t> sog_hard, sog_bits;
    uint64_t sog_target_delta = 0;   // syndrome the flipped positions must produce
    double sog_not_queried = 1.0;    // probability mass of patterns not queried yet
    double sog_list_mass = 0.0;
    double sog_scale = 1.0;          // 2^-rank: fraction of patterns that are syndrome-consistent
    long long sog_queries = 0;

    // Records one queried pattern; returns true when list decoding should stop.
    bool sogrand_query(uint64_t flip_syndrome, double cost, int weight) {
        sog_queries++;
        const double prob = std::exp(-cost);
        sog_not_queried -= prob;
        if (flip_syndrome == sog_target_delta) {
            sog_list_ranks.insert(sog_list_ranks.end(), orb_choice.begin(), orb_choice.begin() + weight);
            sog_list_ptr.push_back((int) sog_list_ranks.size());
            sog_list_prob.push_back(prob);
            sog_list_mass += prob;
            const double not_in_list = sog_scale * std::max(sog_not_queried, 0.0);
            const double confidence = sog_list_mass / (sog_list_mass + not_in_list);
            if (confidence > 1.0 - sogrand_threshold || (int) sog_list_prob.size() >= sogrand_list_size)
                return true;
        }
        return sogrand_max_queries > 0 && sog_queries >= sogrand_max_queries;
    }

    // Queries every set of `weight` distinct ranks (1..n, increasing) with rank sum `rank_sum`,
    // choosing slot `slot` onwards from ranks >= `lowest`, in lexicographic order.
    bool orb_subsets(int slot, int weight, int lowest, long long rank_sum, uint64_t flip_syndrome,
                     double cost) {
        const int n = (int) orb_reliability.size();
        const int rest = weight - slot - 1;
        if (rest == 0) {
            if (rank_sum < lowest || rank_sum > n) return false;
            const int r = (int) rank_sum;
            orb_choice[slot] = r;
            return sogrand_query(flip_syndrome ^ orb_mask[r - 1], cost + orb_reliability[r - 1], weight);
        }
        const long long max_rest = (long long) rest * n - (long long) rest * (rest - 1) / 2;
        const int first = (int) std::max<long long>(lowest, rank_sum - max_rest);
        for (int r = first; r <= n; r++) {
            const long long need = rank_sum - r;
            const long long min_rest = (long long) rest * r + (long long) rest * (rest + 1) / 2;
            if (need < min_rest) break;
            orb_choice[slot] = r;
            if (orb_subsets(slot + 1, weight, r + 1, need, flip_syndrome ^ orb_mask[r - 1],
                            cost + orb_reliability[r - 1]))
                return true;
        }
        return false;
    }

    // true if `syndrome` lies in the span of the GC's column masks (ell <= 63)
    static bool in_column_span(const GeneralizedCheck &gc, uint64_t syndrome) {
        uint64_t basis[64] = {0};
        for (uint64_t v: gc.col_masks) {
            for (int b = 63; b >= 0 && v; b--) {
                if (!((v >> b) & 1ULL)) continue;
                if (!basis[b]) { basis[b] = v; v = 0; }
                else v ^= basis[b];
            }
        }
        for (int b = 63; b >= 0 && syndrome; b--)
            if ((syndrome >> b) & 1ULL) {
                if (!basis[b]) return false;
                syndrome ^= basis[b];
            }
        return true;
    }

    void sogrand_update(const GeneralizedCheck &gc, const std::vector<uint8_t> &synd,
                        const std::vector<double> &lam, std::vector<double> &out) {
        const int s = (int) gc.support.size();
        uint64_t target = 0;
        for (int i = 0; i < gc.ell; i++) if (synd[i]) target |= (1ULL << i);

        sog_hard.resize(s);
        uint64_t hard_syndrome = 0;
        double pm_hard = 0.0;   // -log P(hard decision)
        for (int k = 0; k < s; k++) {
            sog_hard[k] = lam[k] > 0.0 ? 0 : 1;
            if (sog_hard[k]) hard_syndrome ^= gc.col_masks[k];
            pm_hard += std::log1p(std::exp(-std::fabs(lam[k])));
        }
        sog_target_delta = target ^ hard_syndrome;
        if (!in_column_span(gc, sog_target_delta)) {   // inconsistent local syndrome: no information
            std::fill(out.begin(), out.end(), 0.0);
            return;
        }

        // reliability order (least reliable first)
        orb_position.resize(s);
        std::iota(orb_position.begin(), orb_position.end(), 0);
        std::stable_sort(orb_position.begin(), orb_position.end(),
                         [&](int a, int b) { return std::fabs(lam[a]) < std::fabs(lam[b]); });
        orb_reliability.resize(s);
        orb_mask.resize(s);
        for (int r = 0; r < s; r++) {
            orb_reliability[r] = std::fabs(lam[orb_position[r]]);
            orb_mask[r] = gc.col_masks[orb_position[r]];
        }
        int intercept = sogrand_intercept;
        if (intercept < 0) {   // 1-line ORBGRAND: fit the lower half of the sorted reliabilities
            const long long half = std::llround((double) s / 2.0);
            const double slope = half > 1 ? (orb_reliability[half - 1] - orb_reliability[0]) / (double) (half - 1) : 0.0;
            intercept = slope > 0.0 ? (int) std::max<long long>(std::llround(orb_reliability[0] / slope - 1.0), 0) : 0;
        }

        orb_choice.resize(s);
        sog_list_ranks.clear();
        sog_list_ptr.assign(1, 0);
        sog_list_prob.clear();
        sog_not_queried = 1.0;
        sog_list_mass = 0.0;
        sog_scale = std::ldexp(1.0, -gc.rank);
        sog_queries = 0;

        bool stop = sogrand_query(0ULL, pm_hard, 0);
        const long long max_weight = (long long) intercept * s + (long long) s * (s + 1) / 2;
        for (long long wt = intercept + 1; !stop && wt <= max_weight; wt++) {
            for (int w = 1; w <= s; w++) {
                const long long rank_sum = wt - (long long) intercept * w;
                if (rank_sum < (long long) w * (w + 1) / 2) break;
                if (rank_sum > (long long) w * s - (long long) w * (w - 1) / 2) continue;
                if (orb_subsets(0, w, 1, rank_sum, 0ULL, pm_hard)) {
                    stop = true;
                    break;
                }
            }
        }
        candidate_evaluations += sog_queries;

        // bit marginals: list patterns plus the not-in-list mass split by the input marginals
        const double eps = std::numeric_limits<double>::epsilon();
        const double not_in_list = std::max(sog_scale * std::max(sog_not_queried, 0.0), eps);
        sog_p0.assign(s, 0.0);
        sog_p1.assign(s, 0.0);
        for (size_t m = 0; m < sog_list_prob.size(); m++) {
            sog_bits.assign(sog_hard.begin(), sog_hard.end());
            for (int q = sog_list_ptr[m]; q < sog_list_ptr[m + 1]; q++) sog_bits[orb_position[sog_list_ranks[q] - 1]] ^= 1;
            for (int k = 0; k < s; k++) (sog_bits[k] ? sog_p1[k] : sog_p0[k]) += sog_list_prob[m];
        }
        for (int k = 0; k < s; k++) {
            double prior1 = 1.0 / (1.0 + std::exp(lam[k]));
            prior1 = std::min(std::max(prior1, eps), 1.0 - eps);
            const double p0 = sog_p0[k] + not_in_list * (1.0 - prior1);
            const double p1 = sog_p1[k] + not_in_list * prior1;
            out[k] = std::log(p0) - std::log(p1) - lam[k];
        }
    }
};

} // namespace lrbms
} // namespace ldpc

#endif // LDPC_LRBMS_HPP
