#ifndef LDPC_MBP_LRBMS_HPP
#define LDPC_MBP_LRBMS_HPP

/**
 * MBP4 + LRB-MS: quaternary memory-BP variable nodes with LRB-MS generalized checks.
 *
 * CSS code with X checks H_X and Z checks H_Z. Each qubit n carries the quaternary
 * log-ratios Gamma_n^W = ln P(I)/P(W) for W in {X, Y, Z} (index 0, 1, 2).
 *
 * Check layers. A Z-type check only sees whether the error E_n anticommutes with Z, i.e. the
 * x-bit of E_n (E_n in {X, Y}); an X-type check only sees the z-bit (E_n in {Z, Y}). Rows of H_Z
 * are grouped into generalized checks (GCs) acting on x-bits, rows of H_X into GCs acting on
 * z-bits. Each GC is updated with the binary LRB-MS rule of lrbms.hpp, whose output is scaled
 * by mu (the min-sum normalisation).
 *
 * Variable node (MBP4, Kuo and Lai). With Delta_{c->n} the binary output of GC c,
 *     Gamma_n^W = Lambda_n^W + (1 / alpha) * sum_{c : W anticommutes with c's type} Delta_{c->n}.
 * The message to GC c uses Gamma minus c's own previous output, subtracted *without* the
 * 1 / alpha factor (memory / inhibition when alpha < 1):
 *     lambda_{n->c}^W = Gamma_n^W - [W anticommutes] Delta_{c->n},
 * converted to the binary commutation LLR
 *     ln( (1 + e^{-lambda^P}) / (e^{-lambda^{W1}} + e^{-lambda^{W2}}) ),
 * where P commutes with the check type and W1, W2 anticommute.
 *
 * Schedule: serial over GCs (layered: Z-type GCs, then X-type GCs, each iteration) or parallel.
 * With p_Y = p_Z = 0 and alpha = 1, the x-bit decisions coincide with binary LRB-MS on H_Z.
 *
 * The GC update is any method of lrbms.hpp. With gc_method = MAP, mu = 1 / a, alpha = 1 and the
 * parallel schedule this is the generalized MBP4 decoder (GMBP4) of Mostad, Rosnes and Lin with
 * their scaling a: both the posterior and the message to each GC carry the check outputs
 * scaled by 1 / a.
 */

#include "lrbms.hpp"

namespace ldpc {
namespace lrbms {

class MbpLrbmsDecoder {
  public:
    // Pauli indices used for the quaternary log-ratios.
    static constexpr int PAULI_X = 0;
    static constexpr int PAULI_Y = 1;
    static constexpr int PAULI_Z = 2;

    int qubit_count;
    LrbmsDecoder z_layer;               // GCs over rows of H_Z, acting on x-bits
    LrbmsDecoder x_layer;               // GCs over rows of H_X, acting on z-bits
    std::vector<double> channel_llr[3]; // ln P(I)/P(W)
    std::vector<double> posterior[3];   // Gamma^W
    int maximum_iterations;
    double alpha;
    LrbmsSchedule schedule;
    double llr_clip;
    std::vector<uint8_t> error_x; // hard decision: x-part of the error
    std::vector<uint8_t> error_z; // hard decision: z-part of the error
    int iterations = 0;
    bool converge = false;

    MbpLrbmsDecoder(int qubit_count_, const std::vector<std::vector<int>> &hx_rows,
                    const std::vector<std::vector<int>> &hz_rows,
                    const std::vector<std::vector<int>> &x_groups,
                    const std::vector<std::vector<int>> &z_groups, const std::vector<double> &p_x,
                    const std::vector<double> &p_y, const std::vector<double> &p_z, int max_iter,
                    double mu, double alpha_, int lrbms_order, GcMethod gc_method,
                    LrbmsSchedule schedule_, double llr_clip_)
        : qubit_count(qubit_count_),
          z_layer(static_cast<int>(hz_rows.size()), qubit_count_, hz_rows, z_groups,
                  std::vector<double>(qubit_count_, 0.1), max_iter, mu, lrbms_order, gc_method,
                  schedule_, llr_clip_),
          x_layer(static_cast<int>(hx_rows.size()), qubit_count_, hx_rows, x_groups,
                  std::vector<double>(qubit_count_, 0.1), max_iter, mu, lrbms_order, gc_method,
                  schedule_, llr_clip_),
          maximum_iterations(max_iter), alpha(alpha_), schedule(schedule_), llr_clip(llr_clip_) {
        if (alpha <= 0.0) {
            throw std::invalid_argument("MbpLrbmsDecoder: alpha must be positive");
        }
        set_channel(p_x, p_y, p_z);
        error_x.assign(qubit_count, 0);
        error_z.assign(qubit_count, 0);
    }

    void set_channel(const std::vector<double> &p_x, const std::vector<double> &p_y,
                     const std::vector<double> &p_z) {
        if (static_cast<int>(p_x.size()) != qubit_count ||
            static_cast<int>(p_y.size()) != qubit_count ||
            static_cast<int>(p_z.size()) != qubit_count) {
            throw std::invalid_argument("MbpLrbmsDecoder: channel vectors must have length n");
        }
        const std::vector<double> *pauli_probs[3] = {&p_x, &p_y, &p_z};
        for (int w = 0; w < 3; w++) {
            channel_llr[w].assign(qubit_count, 0.0);
        }
        for (int q = 0; q < qubit_count; q++) {
            const double p_identity = 1.0 - p_x[q] - p_y[q] - p_z[q];
            if (p_identity <= 0.0) {
                throw std::invalid_argument("MbpLrbmsDecoder: p_X + p_Y + p_Z must be < 1");
            }
            for (int w = 0; w < 3; w++) {
                const double p_w = (*pauli_probs[w])[q];
                channel_llr[w][q] = p_w <= 0.0 ? llr_clip : std::log(p_identity / p_w);
            }
        }
    }

    void set_mu(double mu) {
        z_layer.ms_scaling_factor = mu;
        x_layer.ms_scaling_factor = mu;
    }

    GcMethod gc_method() const { return z_layer.gc_method; }

    void set_gc_method(GcMethod gc_method) {
        z_layer.gc_method = gc_method;
        x_layer.gc_method = gc_method;
    }

    void set_sogrand(int list_size, double threshold, long long max_queries, int intercept) {
        for (LrbmsDecoder *layer : {&z_layer, &x_layer}) {
            layer->sogrand_list_size = list_size;
            layer->sogrand_threshold = threshold;
            layer->sogrand_max_queries = max_queries;
            layer->sogrand_intercept = intercept;
        }
    }

    // Binary LLRs of the x-part (seen by Z checks) and z-part (seen by X checks) of the
    // current posteriors, e.g. as soft input for OSD on each CSS half.
    void marginal_llrs(std::vector<double> &llr_x, std::vector<double> &llr_z) const {
        llr_x.resize(qubit_count);
        llr_z.resize(qubit_count);
        for (int q = 0; q < qubit_count; q++) {
            const double llr[3] = {posterior[PAULI_X][q], posterior[PAULI_Y][q],
                                   posterior[PAULI_Z][q]};
            llr_x[q] = to_binary_llr(llr, PAULI_Z);
            llr_z[q] = to_binary_llr(llr, PAULI_X);
        }
    }

    // Binary commutation LLR from quaternary log-ratios llr[0..2] (X, Y, Z):
    //     ln(1 + e^{-llr[commuting]}) - ln(e^{-llr[a]} + e^{-llr[b]}),
    // where {a, b} are the two Paulis that anticommute with the check type. Computed stably.
    static double to_binary_llr(const double llr[3], int commuting) {
        const int a = (commuting + 1) % 3;
        const int b = (commuting + 2) % 3;
        const double l_c = llr[commuting];
        const double numerator =
            l_c > 0 ? std::log1p(std::exp(-l_c)) : -l_c + std::log1p(std::exp(l_c));
        const double low = std::min(llr[a], llr[b]);
        const double high = std::max(llr[a], llr[b]);
        const double denominator = -low + std::log1p(std::exp(-(high - low)));
        return numerator - denominator;
    }

    // The Pauli that commutes with every check of a layer: Z for the Z-type layer, X otherwise.
    int commuting_pauli(const LrbmsDecoder &layer) const {
        return &layer == &z_layer ? PAULI_Z : PAULI_X;
    }

    // Updates GC `gc_index` of `layer`. With `parallel_out` == nullptr (serial schedule), the
    // new GC outputs are applied to the posteriors immediately; otherwise they are stored in
    // `parallel_out` and applied later by recompute_posteriors().
    void update_gc(LrbmsDecoder &layer, size_t gc_index, const std::vector<uint8_t> &local_syndrome,
                   std::vector<double> &gc_input, std::vector<double> &gc_output,
                   std::vector<double> *parallel_out) {
        const GeneralizedCheck &gc = layer.gcs[gc_index];
        const int support_size = static_cast<int>(gc.support.size());
        const int commuting = commuting_pauli(layer);
        gc_input.resize(support_size);
        gc_output.resize(support_size);
        for (int k = 0; k < support_size; k++) {
            const int qubit = gc.support[k];
            const double previous = layer.c2v[gc.msg_offset + k];
            double message[3];
            for (int w = 0; w < 3; w++) {
                message[w] = posterior[w][qubit] - (w != commuting ? previous : 0.0);
            }
            gc_input[k] = to_binary_llr(message, commuting);
        }
        layer.gc_update(gc, local_syndrome, gc_input, gc_output);
        for (int k = 0; k < support_size; k++) {
            if (parallel_out != nullptr) {
                (*parallel_out)[gc.msg_offset + k] = gc_output[k];
                continue;
            }
            const int qubit = gc.support[k];
            const double change = (gc_output[k] - layer.c2v[gc.msg_offset + k]) / alpha;
            for (int w = 0; w < 3; w++) {
                if (w != commuting) {
                    posterior[w][qubit] += change;
                }
            }
            layer.c2v[gc.msg_offset + k] = gc_output[k];
        }
    }

    void recompute_posteriors() {
        for (int w = 0; w < 3; w++) {
            posterior[w] = channel_llr[w];
        }
        for (const LrbmsDecoder *layer : {&z_layer, &x_layer}) {
            const int commuting = commuting_pauli(*layer);
            for (int m = 0; m < layer->total_msgs; m++) {
                for (int w = 0; w < 3; w++) {
                    if (w != commuting) {
                        posterior[w][layer->msg_var[m]] += layer->c2v[m] / alpha;
                    }
                }
            }
        }
    }

    static std::vector<std::vector<uint8_t>> local_syndromes(const LrbmsDecoder &layer,
                                                             const std::vector<uint8_t> &syndrome) {
        std::vector<std::vector<uint8_t>> out(layer.gcs.size());
        for (size_t c = 0; c < layer.gcs.size(); c++) {
            out[c].resize(layer.gcs[c].ell);
            for (int i = 0; i < layer.gcs[c].ell; i++) {
                out[c][i] = syndrome[layer.gcs[c].rows[i]] & 1;
            }
        }
        return out;
    }

    void hard_decision() {
        for (int q = 0; q < qubit_count; q++) {
            int best = -1;         // -1 = identity
            double best_llr = 0.0; // the identity has log-ratio 0
            for (int w = 0; w < 3; w++) {
                if (posterior[w][q] < best_llr) {
                    best_llr = posterior[w][q];
                    best = w;
                }
            }
            error_x[q] = (best == PAULI_X || best == PAULI_Y) ? 1 : 0;
            error_z[q] = (best == PAULI_Z || best == PAULI_Y) ? 1 : 0;
        }
    }

    static bool syndrome_matches(const LrbmsDecoder &layer, const std::vector<uint8_t> &bits,
                                 const std::vector<uint8_t> &syndrome) {
        for (int i = 0; i < layer.check_count; i++) {
            uint8_t parity = 0;
            for (int q : layer.pcm_rows[i]) {
                parity ^= bits[q];
            }
            if (parity != (syndrome[i] & 1)) {
                return false;
            }
        }
        return true;
    }

    bool decision_matches(const std::vector<uint8_t> &syndrome_x,
                          const std::vector<uint8_t> &syndrome_z) const {
        return syndrome_matches(z_layer, error_x, syndrome_z) &&
               syndrome_matches(x_layer, error_z, syndrome_x);
    }

    // syndrome_x: syndrome of the X checks (detects the z-part of the error);
    // syndrome_z: syndrome of the Z checks (detects the x-part).
    void decode(const std::vector<uint8_t> &syndrome_x, const std::vector<uint8_t> &syndrome_z) {
        if (static_cast<int>(syndrome_x.size()) != x_layer.check_count ||
            static_cast<int>(syndrome_z.size()) != z_layer.check_count) {
            throw std::invalid_argument("MbpLrbmsDecoder: syndrome length mismatch");
        }
        std::fill(z_layer.c2v.begin(), z_layer.c2v.end(), 0.0);
        std::fill(x_layer.c2v.begin(), x_layer.c2v.end(), 0.0);
        for (int w = 0; w < 3; w++) {
            posterior[w] = channel_llr[w];
        }
        const auto z_local = local_syndromes(z_layer, syndrome_z);
        const auto x_local = local_syndromes(x_layer, syndrome_x);
        const int max_iter = maximum_iterations > 0 ? maximum_iterations : qubit_count;
        converge = false;
        iterations = 0;
        std::vector<double> gc_input;
        std::vector<double> gc_output;

        hard_decision();
        if (decision_matches(syndrome_x, syndrome_z)) {
            converge = true;
            return;
        }
        for (int it = 1; it <= max_iter; it++) {
            iterations = it;
            if (schedule == SERIAL) {
                for (size_t c = 0; c < z_layer.gcs.size(); c++) {
                    update_gc(z_layer, c, z_local[c], gc_input, gc_output, nullptr);
                }
                for (size_t c = 0; c < x_layer.gcs.size(); c++) {
                    update_gc(x_layer, c, x_local[c], gc_input, gc_output, nullptr);
                }
            } else {
                for (size_t c = 0; c < z_layer.gcs.size(); c++) {
                    update_gc(z_layer, c, z_local[c], gc_input, gc_output, &z_layer.c2v_new);
                }
                for (size_t c = 0; c < x_layer.gcs.size(); c++) {
                    update_gc(x_layer, c, x_local[c], gc_input, gc_output, &x_layer.c2v_new);
                }
                z_layer.c2v.swap(z_layer.c2v_new);
                x_layer.c2v.swap(x_layer.c2v_new);
                recompute_posteriors();
            }
            hard_decision();
            if (decision_matches(syndrome_x, syndrome_z)) {
                converge = true;
                break;
            }
        }
    }
};

} // namespace lrbms
} // namespace ldpc

#endif
