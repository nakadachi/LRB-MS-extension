#ifndef LDPC_LRBMS_HPP
#define LDPC_LRBMS_HPP

/**
 * LRB-MS: Least-Reliable-Basis Min-Sum decoding with generalized checks.
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
    TRELLIS = 1
};

enum LrbmsSchedule {
    PARALLEL = 0,
    SERIAL = 1
};

struct GeneralizedCheck {
    std::vector<int> rows;      // global row indices
    std::vector<int> support;   // global column indices (sorted, unique)
    int ell = 0;                // number of rows
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
        if (gc_method == TRELLIS && max_group_size() > max_trellis_ell)
            throw std::invalid_argument("LrbmsDecoder: TRELLIS method limited to groups with ell <= " + std::to_string(max_trellis_ell));

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
        if (gc_method == TRELLIS) trellis_update(gc, synd, in, out);
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
        order_buf.resize(s);
        std::iota(order_buf.begin(), order_buf.end(), 0);
        std::stable_sort(order_buf.begin(), order_buf.end(), [&](int a, int b) { return absl[a] < absl[b]; });

        // reliability-ordered Gaussian elimination -> least reliable basis
        R.assign(gc.row_bits.begin(), gc.row_bits.end());
        sb.assign(synd.begin(), synd.end());
        is_pivot.assign(s, 0);
        pivot_of_row.assign(L, -1);
        int r = 0;
        for (int idx = 0; idx < s && r < L; idx++) {
            int k = order_buf[idx];
            int w = k >> 6;
            uint64_t bit = 1ULL << (k & 63);
            int piv = -1;
            for (int i = r; i < L; i++) if (R[(size_t) i * W + w] & bit) { piv = i; break; }
            if (piv < 0) continue;
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

        best0.assign(s, INF);
        best1.assign(s, INF);
        cur.assign(e0.begin(), e0.end());

        auto score_current = [&](double cost) {
            candidate_evaluations++;
            for (int k = 0; k < s; k++) {
                double ext = cost - (cur[k] != hard[k] ? absl[k] : 0.0);
                if (cur[k]) { if (ext < best1[k]) best1[k] = ext; }
                else { if (ext < best0[k]) best0[k] = ext; }
            }
        };
        auto apply_diff = [&](double &cost) {
            for (int k: diff) {
                cost += (cur[k] != hard[k]) ? -absl[k] : absl[k];
                cur[k] ^= 1;
            }
        };
        auto revert_diff = [&]() { for (int k: diff) cur[k] ^= 1; };

        score_current(cost0);

        // non-pivot positions in increasing reliability
        nonpivots.clear();
        for (int idx = 0; idx < s; idx++) if (!is_pivot[order_buf[idx]]) nonpivots.push_back(order_buf[idx]);

        // order-1 re-encodings
        for (int j: nonpivots) {
            diff.clear();
            diff.push_back(j);
            uint64_t bit = 1ULL << (j & 63);
            int w = j >> 6;
            for (int i = 0; i < r; i++) if (R[(size_t) i * W + w] & bit) diff.push_back(pivot_of_row[i]);
            double cost = cost0;
            apply_diff(cost);
            score_current(cost);
            revert_diff();
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
                double cost = cost0;
                apply_diff(cost);
                score_current(cost);
                revert_diff();
            }
        }

        for (int k = 0; k < s; k++) {
            bool f0 = best0[k] < INF, f1 = best1[k] < INF;
            if (f0 && f1) out[k] = best1[k] - best0[k];
            else if (f0) out[k] = llr_clip;   // bit forced to 0 by the local syndrome
            else if (f1) out[k] = -llr_clip;  // bit forced to 1
            else out[k] = 0.0;
        }
    }

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
};

} // namespace lrbms
} // namespace ldpc

#endif // LDPC_LRBMS_HPP
