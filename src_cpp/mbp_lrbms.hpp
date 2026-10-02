#ifndef LDPC_MBP_LRBMS_HPP
#define LDPC_MBP_LRBMS_HPP

/**
 * MBP4 + LRB-MS: quaternary memory-BP variable nodes with LRB-MS generalized checks.
 *
 * CSS code with X checks H_X and Z checks H_Z. Each qubit n carries the quaternary
 * log-ratios Gamma_n^W = ln P(I)/P(W) for W in {X, Y, Z} (index 0, 1, 2).
 *
 * Check layers. A Z-type check only sees whether E_n anticommutes with Z, i.e. the x-bit of
 * E_n (E_n in {X, Y}); an X-type check only sees the z-bit (E_n in {Z, Y}). Rows of H_Z are
 * grouped into generalized checks (GCs) that act on x-bits, rows of H_X into GCs acting on
 * z-bits. Each GC is updated with the binary LRB-MS rule of lrbms.hpp, whose output is
 * scaled by mu (the min-sum normalisation).
 *
 * Variable node (MBP4, Kuo & Lai). With Delta_{c->n} the binary output of GC c,
 *     Gamma_n^W = Lambda_n^W + (1/alpha) * sum_{c : W anticommutes with c's type} Delta_{c->n}.
 * The message to GC c uses Gamma minus c's own previous output, subtracted *without* the
 * 1/alpha factor (memory / inhibition when alpha < 1):
 *     lambda_{n->c}^W = Gamma_n^W - [W anticommutes] Delta_{c->n},
 * converted to the binary commutation LLR
 *     ln( (1 + e^{-lambda^P}) / (e^{-lambda^{W1}} + e^{-lambda^{W2}}) ),
 * where P commutes with the check type and W1, W2 anticommute.
 *
 * Schedule: serial over GCs (layered; Z-type GCs, then X-type GCs, each iteration) or parallel.
 * With p_Y = p_Z = 0 and alpha = 1, the x-bit decisions coincide with binary LRB-MS on H_Z.
 */

#include "lrbms.hpp"

namespace ldpc {
namespace lrbms {

class MbpLrbmsDecoder {
public:
    int n;
    LrbmsDecoder zlayer;   // GCs over rows of H_Z, acting on x-bits
    LrbmsDecoder xlayer;   // GCs over rows of H_X, acting on z-bits
    std::vector<double> lam[3];     // channel log-ratios ln P(I)/P(W)
    std::vector<double> gamma[3];   // posteriors
    int maximum_iterations;
    double alpha;
    LrbmsSchedule schedule;
    double llr_clip;
    std::vector<uint8_t> ex, ez;    // hard decision: x-part and z-part of the error
    int iterations = 0;
    bool converge = false;

    MbpLrbmsDecoder(int n_, const std::vector<std::vector<int>> &hx_rows, const std::vector<std::vector<int>> &hz_rows,
                    const std::vector<std::vector<int>> &x_groups, const std::vector<std::vector<int>> &z_groups,
                    const std::vector<double> &px, const std::vector<double> &py, const std::vector<double> &pz,
                    int max_iter, double mu, double alpha_, int order, LrbmsSchedule sched, double clip)
        : n(n_),
          zlayer((int) hz_rows.size(), n_, hz_rows, z_groups, std::vector<double>(n_, 0.1), max_iter, mu, order, LRBMS, sched, clip),
          xlayer((int) hx_rows.size(), n_, hx_rows, x_groups, std::vector<double>(n_, 0.1), max_iter, mu, order, LRBMS, sched, clip),
          maximum_iterations(max_iter), alpha(alpha_), schedule(sched), llr_clip(clip) {
        if (alpha <= 0.0) throw std::invalid_argument("MbpLrbmsDecoder: alpha must be positive");
        set_channel(px, py, pz);
        ex.assign(n, 0);
        ez.assign(n, 0);
    }

    void set_channel(const std::vector<double> &px, const std::vector<double> &py, const std::vector<double> &pz) {
        if ((int) px.size() != n || (int) py.size() != n || (int) pz.size() != n)
            throw std::invalid_argument("MbpLrbmsDecoder: channel vectors must have length n");
        const std::vector<double> *pw[3] = {&px, &py, &pz};
        for (int w = 0; w < 3; w++) lam[w].assign(n, 0.0);
        for (int j = 0; j < n; j++) {
            double pi = 1.0 - px[j] - py[j] - pz[j];
            if (pi <= 0.0) throw std::invalid_argument("MbpLrbmsDecoder: p_X + p_Y + p_Z must be < 1");
            for (int w = 0; w < 3; w++) {
                double p = (*pw[w])[j];
                lam[w][j] = p <= 0.0 ? llr_clip : std::log(pi / p);
            }
        }
    }

    void set_mu(double mu) { zlayer.ms_scaling_factor = mu; xlayer.ms_scaling_factor = mu; }

    // binary commutation LLR from quaternary message log-ratios l[0..2] (X, Y, Z);
    // c = index of the Pauli that commutes with the check type (Z-type check: Z = 2; X-type: X = 0)
    static double to_binary(const double l[3], int c) {
        // ln(1 + e^{-l_c}) - ln(e^{-l_a} + e^{-l_b}) for {a, b} the other two, computed stably
        int a = (c + 1) % 3, b = (c + 2) % 3;
        double num = l[c] > 0 ? std::log1p(std::exp(-l[c])) : -l[c] + std::log1p(std::exp(l[c]));
        double m = std::min(l[a], l[b]);
        double den = -m + std::log1p(std::exp(-(std::max(l[a], l[b]) - m)));
        return num - den;
    }

    // layer type: 0 = Z-type GCs (anticommuting Paulis X, Y; commuting Z), 1 = X-type (anticommuting Z, Y; commuting X)
    void update_gc(LrbmsDecoder &L, size_t c, int type, const std::vector<uint8_t> &synd,
                   std::vector<double> &in, std::vector<double> &out, std::vector<double> *new_msgs) {
        auto &gc = L.gcs[c];
        const int s = (int) gc.support.size();
        const int comm = type == 0 ? 2 : 0;
        in.resize(s);
        out.resize(s);
        for (int k = 0; k < s; k++) {
            int j = gc.support[k];
            double old = L.c2v[gc.msg_offset + k];
            double l[3];
            for (int w = 0; w < 3; w++) l[w] = gamma[w][j] - (w != comm ? old : 0.0);
            in[k] = to_binary(l, comm);
        }
        L.gc_update(gc, synd, in, out);
        for (int k = 0; k < s; k++) {
            if (new_msgs) { (*new_msgs)[gc.msg_offset + k] = out[k]; continue; }
            int j = gc.support[k];
            double delta = (out[k] - L.c2v[gc.msg_offset + k]) / alpha;
            for (int w = 0; w < 3; w++) if (w != comm) gamma[w][j] += delta;
            L.c2v[gc.msg_offset + k] = out[k];
        }
    }

    void recompute_gamma() {
        for (int w = 0; w < 3; w++) gamma[w] = lam[w];
        for (int type = 0; type < 2; type++) {
            LrbmsDecoder &L = type == 0 ? zlayer : xlayer;
            const int comm = type == 0 ? 2 : 0;
            for (int t = 0; t < L.total_msgs; t++)
                for (int w = 0; w < 3; w++) if (w != comm) gamma[w][L.msg_var[t]] += L.c2v[t] / alpha;
        }
    }

    static std::vector<std::vector<uint8_t>> local_syndromes(const LrbmsDecoder &L, const std::vector<uint8_t> &s) {
        std::vector<std::vector<uint8_t>> out(L.gcs.size());
        for (size_t c = 0; c < L.gcs.size(); c++) {
            out[c].resize(L.gcs[c].ell);
            for (int i = 0; i < L.gcs[c].ell; i++) out[c][i] = s[L.gcs[c].rows[i]] & 1;
        }
        return out;
    }

    void hard_decision() {
        for (int j = 0; j < n; j++) {
            int best = -1;
            double mn = 0.0;   // identity has log-ratio 0
            for (int w = 0; w < 3; w++) if (gamma[w][j] < mn) { mn = gamma[w][j]; best = w; }
            ex[j] = (best == 0 || best == 1) ? 1 : 0;   // X or Y
            ez[j] = (best == 2 || best == 1) ? 1 : 0;   // Z or Y
        }
    }

    bool matches(const LrbmsDecoder &L, const std::vector<uint8_t> &bits, const std::vector<uint8_t> &s) const {
        for (int i = 0; i < L.check_count; i++) {
            uint8_t b = 0;
            for (int j: L.pcm_rows[i]) b ^= bits[j];
            if (b != (s[i] & 1)) return false;
        }
        return true;
    }

    // sx: syndrome of the X checks (detects z-part), sz: syndrome of the Z checks (detects x-part)
    void decode(const std::vector<uint8_t> &sx, const std::vector<uint8_t> &sz) {
        if ((int) sx.size() != xlayer.check_count || (int) sz.size() != zlayer.check_count)
            throw std::invalid_argument("MbpLrbmsDecoder: syndrome length mismatch");
        std::fill(zlayer.c2v.begin(), zlayer.c2v.end(), 0.0);
        std::fill(xlayer.c2v.begin(), xlayer.c2v.end(), 0.0);
        for (int w = 0; w < 3; w++) gamma[w] = lam[w];
        auto zs = local_syndromes(zlayer, sz);
        auto xs = local_syndromes(xlayer, sx);
        int max_iter = maximum_iterations > 0 ? maximum_iterations : n;
        converge = false;
        iterations = 0;
        std::vector<double> in, out;
        hard_decision();
        if (matches(zlayer, ex, sz) && matches(xlayer, ez, sx)) { converge = true; return; }
        for (int it = 1; it <= max_iter; it++) {
            iterations = it;
            if (schedule == SERIAL) {
                for (size_t c = 0; c < zlayer.gcs.size(); c++) update_gc(zlayer, c, 0, zs[c], in, out, nullptr);
                for (size_t c = 0; c < xlayer.gcs.size(); c++) update_gc(xlayer, c, 1, xs[c], in, out, nullptr);
            } else {
                for (size_t c = 0; c < zlayer.gcs.size(); c++) update_gc(zlayer, c, 0, zs[c], in, out, &zlayer.c2v_new);
                for (size_t c = 0; c < xlayer.gcs.size(); c++) update_gc(xlayer, c, 1, xs[c], in, out, &xlayer.c2v_new);
                zlayer.c2v.swap(zlayer.c2v_new);
                xlayer.c2v.swap(xlayer.c2v_new);
                recompute_gamma();
            }
            hard_decision();
            if (matches(zlayer, ex, sz) && matches(xlayer, ez, sx)) { converge = true; break; }
        }
    }
};

} // namespace lrbms
} // namespace ldpc

#endif
