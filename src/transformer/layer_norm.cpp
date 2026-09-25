#include "grad/transformer/tensor.h"
#include "grad/transformer/layer_norm.h"
#include "grad/transformer/parallel.h"
#include "grad/utils/narrow.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace grad {

LayerNorm::LayerNorm(int d_model, bool rms) : d_model_(d_model), epsilon(1e-5f), rms_(rms) {
    const size_t d = narrow<size_t>(d_model);
    Tensor gamma_tensor(1, d);
    Tensor beta_tensor(1, d);
    gamma_tensor.fill(1.0f);  // beta stays at the constructor's zeros

    gamma = Variable::create(gamma_tensor, true);
    beta = Variable::create(beta_tensor, true);
}

// Normalizes over the innermost dimension. Rows are contiguous at any
// rank, so a 2D (rows, d) and a 3D (batch, seq, d) input are the same loop
// over the flat rows.
std::shared_ptr<Variable> LayerNorm::forward(std::shared_ptr<Variable> input) const {
    const Tensor& input_tensor = input->getData();
    if (input_tensor.getCols() != static_cast<size_t>(d_model_)) {
        throw std::invalid_argument("LayerNorm: input " + input_tensor.shape().to_string()
                                    + " does not end in d_model " + std::to_string(d_model_));
    }

    const size_t total_rows = input_tensor.getFlatRows();
    Tensor result = Tensor::empty_like(input_tensor);

    const float* input_data = input_tensor.raw();
    const float* gamma_data = gamma->getData().raw();
    const float* beta_data = beta->getData().raw();
    float* result_data = result.raw();

    // Per-row statistics are kept for the backward pass so it does not
    // recompute them.
    std::vector<float> means(total_rows);
    std::vector<float> inv_stds(total_rows);

    const size_t d = input_tensor.getCols();
    const float df = static_cast<float>(d);
    const float eps = epsilon;
    const bool rms = rms_;

    parallel_for(total_rows, 16, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; i++) {
            const float* row_in = input_data + i * d;
            float* row_out = result_data + i * d;

            if (rms) {
                // RMSNorm: no mean subtraction, no beta. inv_stds carries
                // 1/rms for the backward pass; means stays zero.
                float ms = 0.0f;
                for (size_t j = 0; j < d; j++) {
                    ms += row_in[j] * row_in[j];
                }
                const float r = 1.0f / std::sqrt(ms / df + eps);
                means[i] = 0.0f;
                inv_stds[i] = r;
                for (size_t j = 0; j < d; j++) {
                    row_out[j] = gamma_data[j] * row_in[j] * r;
                }
                continue;
            }

            float mean = 0.0f;
            for (size_t j = 0; j < d; j++) {
                mean += row_in[j];
            }
            mean /= df;
            means[i] = mean;

            float variance = 0.0f;
            for (size_t j = 0; j < d; j++) {
                float diff = row_in[j] - mean;
                variance += diff * diff;
            }
            variance /= df;

            const float std_inv = 1.0f / std::sqrt(variance + eps);
            inv_stds[i] = std_inv;

            for (size_t j = 0; j < d; j++) {
                float norm = (row_in[j] - mean) * std_inv;
                row_out[j] = gamma_data[j] * norm + beta_data[j];
            }
        }
    });

    const bool needs_grad = compute_requires_grad(input, gamma, beta);
    auto output = Variable::create(std::move(result), needs_grad);

    if (needs_grad) {
        auto self_input = input;
        auto self_gamma = gamma;
        auto self_beta = beta;

        output->setBackward({input, gamma, beta},
                            [self_input, self_gamma, self_beta, means, inv_stds, d, df, total_rows,
                             rms](Variable& node) {
            // Each gradient is computed only if its target requires grad
            // (beta never does in RMS mode): ensureGrad leaves a frozen
            // tensor's grad unallocated, so it must not be written.
            const bool grad_gamma = self_gamma->requiresGrad();
            const bool grad_beta = !rms && self_beta->requiresGrad();
            const bool grad_input = self_input->requiresGrad();
            if (grad_gamma) self_gamma->ensureGrad();
            if (grad_beta) self_beta->ensureGrad();
            if (grad_input) self_input->ensureGrad();

            const float* output_grad_data = node.getGrad().raw();
            const float* gamma_vals = self_gamma->getData().raw();
            const float* input_vals = self_input->getData().raw();
            float* dInput_out = grad_input ? self_input->getGrad().raw() : nullptr;

            // dInput rows are disjoint across the parallel chunks, but
            // dGamma/dBeta sum over every row. Each fixed block of
            // kRowsPerBlock rows accumulates its own partial sums, and the
            // partials are added in block order afterwards: float addition
            // is not associative, so merging in completion order (as a
            // lock would) makes the gradient depend on thread timing. This
            // way it is bit-identical for any thread count or schedule.
            constexpr size_t kRowsPerBlock = 16;
            const size_t num_blocks = (total_rows + kRowsPerBlock - 1) / kRowsPerBlock;
            std::vector<float> g_parts(grad_gamma ? num_blocks * d : 0, 0.0f);
            std::vector<float> b_parts(grad_beta ? num_blocks * d : 0, 0.0f);

            parallel_for(num_blocks, 1, [&](size_t block_begin, size_t block_end) {
                for (size_t blk = block_begin; blk < block_end; blk++) {
                    float* g_part = grad_gamma ? g_parts.data() + blk * d : nullptr;
                    float* b_part = grad_beta ? b_parts.data() + blk * d : nullptr;
                    const size_t begin = blk * kRowsPerBlock;
                    const size_t end = std::min(begin + kRowsPerBlock, total_rows);

                    if (rms) {
                        // y_j = g_j * x_j * r with r = 1/sqrt(mean(x^2)+eps):
                        //   dg_j += dy_j * x_j * r
                        //   dx_j += g_j*dy_j*r - x_j * r^3/d * sum_k(dy_k*g_k*x_k)
                        for (size_t i = begin; i < end; i++) {
                            const float r = inv_stds[i];
                            const float* dout_row = output_grad_data + i * d;
                            const float* input_row = input_vals + i * d;

                            if (g_part) {
                                for (size_t j = 0; j < d; j++) {
                                    g_part[j] += dout_row[j] * input_row[j] * r;
                                }
                            }
                            if (!dInput_out) continue;

                            float dot = 0.0f;
                            for (size_t j = 0; j < d; j++) {
                                dot += dout_row[j] * gamma_vals[j] * input_row[j];
                            }
                            const float k = dot * r * r * r / df;
                            float* dInput_row = dInput_out + i * d;
                            for (size_t j = 0; j < d; j++) {
                                dInput_row[j] += gamma_vals[j] * dout_row[j] * r - input_row[j] * k;
                            }
                        }
                        continue;
                    }

                    for (size_t i = begin; i < end; i++) {
                        const float std_inv = inv_stds[i];
                        const float mean = means[i];
                        const float* dout_row = output_grad_data + i * d;
                        const float* input_row = input_vals + i * d;

                        if (g_part || b_part) {
                            for (size_t j = 0; j < d; j++) {
                                const float x_minus_mean = input_row[j] - mean;
                                const float normalized_ij = x_minus_mean * std_inv;
                                if (g_part) g_part[j] += dout_row[j] * normalized_ij;
                                if (b_part) b_part[j] += dout_row[j];
                            }
                        }
                        if (!dInput_out) continue;

                        // d(var)^(-1/2)/d(var) = -0.5 * (var + eps)^(-3/2) = -0.5 * std_inv^3,
                        // one multiply per row instead of a pow per element.
                        const float dvar_scale = -0.5f * std_inv * std_inv * std_inv;
                        float dvar = 0.0f;
                        for (size_t j = 0; j < d; j++) {
                            const float x_minus_mean = input_row[j] - mean;
                            const float dnorm = dout_row[j] * gamma_vals[j];
                            dvar += dnorm * x_minus_mean * dvar_scale;
                        }

                        float dmean = 0.0f;
                        for (size_t j = 0; j < d; j++) {
                            const float dnorm = dout_row[j] * gamma_vals[j];
                            const float x_minus_mean = input_row[j] - mean;
                            dmean += dnorm * -std_inv + dvar * -2.0f * x_minus_mean / df;
                        }

                        float* dInput_row = dInput_out + i * d;
                        for (size_t j = 0; j < d; j++) {
                            const float dnorm = dout_row[j] * gamma_vals[j];
                            const float x_minus_mean = input_row[j] - mean;
                            dInput_row[j] +=
                                dnorm * std_inv + dvar * 2.0f * x_minus_mean / df + dmean / df;
                        }
                    }
                }
            });

            if (grad_gamma) {
                float* dGamma_out = self_gamma->getGrad().raw();
                for (size_t blk = 0; blk < num_blocks; blk++) {
                    const float* g_part = g_parts.data() + blk * d;
                    for (size_t j = 0; j < d; j++) dGamma_out[j] += g_part[j];
                }
            }
            if (grad_beta) {
                float* dBeta_out = self_beta->getGrad().raw();
                for (size_t blk = 0; blk < num_blocks; blk++) {
                    const float* b_part = b_parts.data() + blk * d;
                    for (size_t j = 0; j < d; j++) dBeta_out[j] += b_part[j];
                }
            }
        });
    }

    return output;
}

}  // namespace grad
