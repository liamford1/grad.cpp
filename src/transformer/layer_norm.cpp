#include "transformer/tensor.h"
#include "transformer/layer_norm.h"
#include "transformer/parallel.h"
#include <cmath>
#include <mutex>
#include <vector>

LayerNorm::LayerNorm(int d_model, bool rms) :
    d_model(d_model),
    epsilon(1e-5f),
    rms_(rms) {

    Tensor gamma_tensor(1, d_model);
    Tensor beta_tensor(1, d_model);
    gamma_tensor.fill(1.0f);
    beta_tensor.fill(0.0f);

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

    auto output = Variable::create(std::move(result), input->requiresGrad());

    if (input->requiresGrad()) {
        auto self_input = input;
        auto self_gamma = gamma;
        auto self_beta = beta;
        int self_d = d_model;
        float self_epsilon = epsilon;

        output->addChild(input);
        output->addChild(gamma);
        output->addChild(beta);

        const bool rms = rms_;
        output->setBackwardFn([self_input, self_gamma, self_beta,
                               output_weak = std::weak_ptr<Variable>(output),
                               means, inv_stds, self_d, self_epsilon, total_rows, rms]() {
            auto output = output_weak.lock();
            if (!output || !output->hasGrad()) return;
            self_gamma->ensureGrad();
            if (!rms) self_beta->ensureGrad();
            self_input->ensureGrad();

            const float* output_grad_data = output->getGrad().raw();
            const float* gamma_data = self_gamma->getData().raw();
            const float* input_data = self_input->getData().raw();
            float* dGamma_out = self_gamma->getGrad().raw();
            float* dBeta_out = rms ? nullptr : self_beta->getGrad().raw();
            float* dInput_out = self_input->getGrad().raw();

            // dInput rows are disjoint across the parallel chunks, but
            // dGamma/dBeta sum over every row: each chunk accumulates
            // private partials and merges them under a lock once.
            std::mutex merge_mutex;

            parallel_for(total_rows, 16, [&](size_t begin, size_t end) {
                std::vector<float> g_part(self_d, 0.0f);
                std::vector<float> b_part(self_d, 0.0f);

                if (rms) {
                    // y_j = g_j * x_j * r with r = 1/sqrt(mean(x^2)+eps):
                    //   dg_j += dy_j * x_j * r
                    //   dx_j += g_j*dy_j*r - x_j * r^3/d * sum_k(dy_k*g_k*x_k)
                    for (size_t i = begin; i < end; i++) {
                        const float r = inv_stds[i];
                        const float* dout_row = output_grad_data + i * self_d;
                        const float* input_row = input_data + i * self_d;
                        float* dInput_row = dInput_out + i * self_d;

                        float dot = 0.0f;
                        for (int j = 0; j < self_d; j++) {
                            g_part[j] += dout_row[j] * input_row[j] * r;
                            dot += dout_row[j] * gamma_data[j] * input_row[j];
                        }
                        const float k = dot * r * r * r / self_d;
                        for (int j = 0; j < self_d; j++) {
                            dInput_row[j] += gamma_data[j] * dout_row[j] * r
                                           - input_row[j] * k;
                        }
                    }
                    std::lock_guard<std::mutex> lk(merge_mutex);
                    for (int j = 0; j < self_d; j++) {
                        dGamma_out[j] += g_part[j];
                    }
                    return;
                }

                for (size_t i = begin; i < end; i++) {
                    const float std_inv = inv_stds[i];
                    const float variance = (1.0f / (std_inv * std_inv)) - self_epsilon;
                    const float mean = means[i];
                    const float* dout_row = output_grad_data + i * self_d;
                    const float* input_row = input_data + i * self_d;

                    float dvar = 0.0f;
                    for (int j = 0; j < self_d; j++) {
                        const float x_minus_mean = input_row[j] - mean;
                        const float normalized_ij = x_minus_mean * std_inv;

                        g_part[j] += dout_row[j] * normalized_ij;
                        b_part[j] += dout_row[j];

                        const float dnorm = dout_row[j] * gamma_data[j];
                        dvar += dnorm * x_minus_mean * -0.5f * std::pow(variance + self_epsilon, -1.5f);
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

                std::lock_guard<std::mutex> lk(merge_mutex);
                for (int j = 0; j < self_d; j++) {
                    dGamma_out[j] += g_part[j];
                    dBeta_out[j] += b_part[j];
                }
            });
        });
    }

    return output;
}
