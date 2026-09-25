#include "transformer/tensor.h"
#include "transformer/layer_norm.h"
#include "transformer/parallel.h"
#include <cmath>
#include <algorithm>
#include <vector>

LayerNorm::LayerNorm(int d_model, bool rms) :
    d_model(d_model),
    epsilon(1e-5f),
    rms_(rms) {

    Tensor gamma_tensor(1, d_model);
    Tensor beta_tensor(1, d_model);
    gamma_tensor.fill(1.0f);  // beta stays at the constructor's zeros

    gamma = Variable::create(gamma_tensor, true);
    beta = Variable::create(beta_tensor, true);
}


// Rows are contiguous whether the input is 2D (rows, d) or 3D
// (batch, seq, d), so both cases are one loop over batch*rows.
std::shared_ptr<Variable> LayerNorm::forward(std::shared_ptr<Variable> input) const {
    const Tensor& input_tensor = input->getData();

    const int total_rows = input_tensor.getIs3D()
        ? static_cast<int>(input_tensor.getBatchSize() * input_tensor.getRows())
        : static_cast<int>(input_tensor.getRows());

    Tensor result = input_tensor.getIs3D()
        ? Tensor::uninitialized(input_tensor.getBatchSize(), input_tensor.getRows(), d_model)
        : Tensor::uninitialized(input_tensor.getRows(), d_model);

    const float* input_data = input_tensor.raw();
    const float* gamma_data = gamma->getData().raw();
    const float* beta_data = beta->getData().raw();
    float* result_data = result.raw();

    // Per-row statistics are kept for the backward pass so it does not
    // recompute them.
    std::vector<float> means(total_rows);
    std::vector<float> inv_stds(total_rows);

    const int d = d_model;
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
                for (int j = 0; j < d; j++) {
                    ms += row_in[j] * row_in[j];
                }
                const float r = 1.0f / std::sqrt(ms / d + eps);
                means[i] = 0.0f;
                inv_stds[i] = r;
                for (int j = 0; j < d; j++) {
                    row_out[j] = gamma_data[j] * row_in[j] * r;
                }
                continue;
            }

            float mean = 0.0f;
            for (int j = 0; j < d; j++) {
                mean += row_in[j];
            }
            mean /= d;
            means[i] = mean;

            float variance = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = row_in[j] - mean;
                variance += diff * diff;
            }
            variance /= d;

            const float std_inv = 1.0f / std::sqrt(variance + eps);
            inv_stds[i] = std_inv;

            for (int j = 0; j < d; j++) {
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
        int self_d = d_model;

        output->addChild(input);
        output->addChild(gamma);
        output->addChild(beta);

        const bool rms = rms_;
        output->setBackwardFn([self_input, self_gamma, self_beta,
                               output_weak = std::weak_ptr<Variable>(output),
                               means, inv_stds, self_d, total_rows, rms]() {
            auto output = output_weak.lock();
            if (!output || !output->hasGrad()) return;

            // Each gradient is computed only if its target requires grad
            // (beta never does in RMS mode): ensureGrad leaves a frozen
            // tensor's grad unallocated, so it must not be written.
            const bool grad_gamma = self_gamma->requiresGrad();
            const bool grad_beta = !rms && self_beta->requiresGrad();
            const bool grad_input = self_input->requiresGrad();
            if (grad_gamma) self_gamma->ensureGrad();
            if (grad_beta) self_beta->ensureGrad();
            if (grad_input) self_input->ensureGrad();

            const float* output_grad_data = output->getGrad().raw();
            const float* gamma_data = self_gamma->getData().raw();
            const float* input_data = self_input->getData().raw();
            float* dInput_out = grad_input ? self_input->getGrad().raw() : nullptr;

            // dInput rows are disjoint across the parallel chunks, but
            // dGamma/dBeta sum over every row. Each fixed block of
            // kRowsPerBlock rows accumulates its own partial sums, and the
            // partials are added in block order afterwards: float addition
            // is not associative, so merging in completion order (as a
            // lock would) makes the gradient depend on thread timing. This
            // way it is bit-identical for any thread count or schedule.
            constexpr size_t kRowsPerBlock = 16;
            const size_t d = static_cast<size_t>(self_d);
            const size_t num_blocks = (total_rows + kRowsPerBlock - 1) / kRowsPerBlock;
            std::vector<float> g_parts(grad_gamma ? num_blocks * d : 0, 0.0f);
            std::vector<float> b_parts(grad_beta ? num_blocks * d : 0, 0.0f);

            parallel_for(num_blocks, 1, [&](size_t block_begin, size_t block_end) {
                for (size_t blk = block_begin; blk < block_end; blk++) {
                    float* g_part = grad_gamma ? g_parts.data() + blk * d : nullptr;
                    float* b_part = grad_beta ? b_parts.data() + blk * d : nullptr;
                    const size_t begin = blk * kRowsPerBlock;
                    const size_t end = std::min(begin + kRowsPerBlock, static_cast<size_t>(total_rows));

                    if (rms) {
                        // y_j = g_j * x_j * r with r = 1/sqrt(mean(x^2)+eps):
                        //   dg_j += dy_j * x_j * r
                        //   dx_j += g_j*dy_j*r - x_j * r^3/d * sum_k(dy_k*g_k*x_k)
                        for (size_t i = begin; i < end; i++) {
                            const float r = inv_stds[i];
                            const float* dout_row = output_grad_data + i * self_d;
                            const float* input_row = input_data + i * self_d;

                            if (g_part) {
                                for (int j = 0; j < self_d; j++) {
                                    g_part[j] += dout_row[j] * input_row[j] * r;
                                }
                            }
                            if (!dInput_out) continue;

                            float dot = 0.0f;
                            for (int j = 0; j < self_d; j++) {
                                dot += dout_row[j] * gamma_data[j] * input_row[j];
                            }
                            const float k = dot * r * r * r / self_d;
                            float* dInput_row = dInput_out + i * self_d;
                            for (int j = 0; j < self_d; j++) {
                                dInput_row[j] += gamma_data[j] * dout_row[j] * r
                                               - input_row[j] * k;
                            }
                        }
                        continue;
                    }

                    for (size_t i = begin; i < end; i++) {
                        const float std_inv = inv_stds[i];
                        const float mean = means[i];
                        const float* dout_row = output_grad_data + i * self_d;
                        const float* input_row = input_data + i * self_d;

                        if (g_part || b_part) {
                            for (int j = 0; j < self_d; j++) {
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
                        for (int j = 0; j < self_d; j++) {
                            const float x_minus_mean = input_row[j] - mean;
                            const float dnorm = dout_row[j] * gamma_data[j];
                            dvar += dnorm * x_minus_mean * dvar_scale;
                        }

                        float dmean = 0.0f;
                        for (int j = 0; j < self_d; j++) {
                            const float dnorm = dout_row[j] * gamma_data[j];
                            const float x_minus_mean = input_row[j] - mean;
                            dmean += dnorm * -std_inv + dvar * -2.0f * x_minus_mean / self_d;
                        }

                        float* dInput_row = dInput_out + i * self_d;
                        for (int j = 0; j < self_d; j++) {
                            const float dnorm = dout_row[j] * gamma_data[j];
                            const float x_minus_mean = input_row[j] - mean;
                            dInput_row[j] += dnorm * std_inv + dvar * 2.0f * x_minus_mean / self_d + dmean / self_d;
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
