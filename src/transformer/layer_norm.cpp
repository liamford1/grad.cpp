#include "transformer/tensor.h"
#include "transformer/layer_norm.h"
#include <iostream>
#include <cmath>

LayerNorm::LayerNorm(int d_model) : 
    d_model(d_model),
    epsilon(1e-5f) {
    
    Tensor gamma_tensor(1, d_model);
    Tensor beta_tensor(1, d_model);
    gamma_tensor.fill(1.0f);
    beta_tensor.fill(0.0f);
    
    gamma = Variable::create(gamma_tensor, true);
    beta = Variable::create(beta_tensor, true);
}

LayerNorm::~LayerNorm() {}

std::shared_ptr<Variable> LayerNorm::forward(std::shared_ptr<Variable> input) const {
    const Tensor& input_tensor = input->getData();

    Tensor result;
    std::vector<float> means;
    std::vector<float> inv_stds;

    if (input_tensor.getDevice() == Device::CUDA) {
        const Tensor& gamma_data = gamma->getData();
        const Tensor& beta_data = beta->getData();
        Tensor gamma_gpu = (gamma_data.getDevice() == Device::CUDA) ? gamma_data : gamma_data.to(Device::CUDA);
        Tensor beta_gpu = (beta_data.getDevice() == Device::CUDA) ? beta_data : beta_data.to(Device::CUDA);

        result = layer_norm_gpu(input_tensor, gamma_gpu, beta_gpu, epsilon, d_model);
        auto output = Variable::create(result, input->requiresGrad());
        if (input->requiresGrad()) {
            auto self_input = input;
            auto self_gamma = gamma;
            auto self_beta = beta;
            int self_d_model = d_model;
            float self_epsilon = epsilon;

            output->addChild(input);
            output->addChild(gamma);
            output->addChild(beta);

            output->setBackwardFn([self_input, self_gamma, self_beta, output_weak = std::weak_ptr<Variable>(output), self_d_model, self_epsilon]() {
                auto output = output_weak.lock();
                if (!output) return;

                Tensor output_grad_cpu = output->getGrad().to(Device::CPU);
                Tensor input_cpu = self_input->getData().to(Device::CPU);
                Tensor gamma_cpu = self_gamma->getData().to(Device::CPU);

                bool is_3d = input_cpu.getIs3D();
                int batch_size = is_3d ? input_cpu.getBatchSize() : 1;
                int rows = is_3d ? input_cpu.getRows() : input_cpu.getRows();
                int total_rows = is_3d ? batch_size * rows : rows;

                std::vector<float> means(total_rows);
                std::vector<float> inv_stds(total_rows);

                const float* input_data = input_cpu.raw();

                if (is_3d) {
                    for (int b = 0; b < batch_size; b++) {
                        const int batch_offset = b * rows * self_d_model;
                        for (int i = 0; i < rows; i++) {
                            const int row_idx = b * rows + i;
                            const float* row_in = input_data + batch_offset + i * self_d_model;
                            float mean = 0.0f;
                            for (int j = 0; j < self_d_model; j++) {
                                mean += row_in[j];
                            }
                            mean /= self_d_model;
                            means[row_idx] = mean;

                            float variance = 0.0f;
                            for (int j = 0; j < self_d_model; j++) {
                                float diff = row_in[j] - mean;
                                variance += diff * diff;
                            }
                            variance /= self_d_model;
                            inv_stds[row_idx] = 1.0f / std::sqrt(variance + self_epsilon);
                        }
                    }
                } else {
                    for (int i = 0; i < rows; i++) {
                        const float* row_in = input_data + i * self_d_model;
                        float mean = 0.0f;
                        for (int j = 0; j < self_d_model; j++) {
                            mean += row_in[j];
                        }
                        mean /= self_d_model;
                        means[i] = mean;

                        float variance = 0.0f;
                        for (int j = 0; j < self_d_model; j++) {
                            float diff = row_in[j] - mean;
                            variance += diff * diff;
                        }
                        variance /= self_d_model;
                        inv_stds[i] = 1.0f / std::sqrt(variance + self_epsilon);
                    }
                }

                Tensor dGamma(1, self_d_model);
                Tensor dBeta(1, self_d_model);
                dGamma.fill(0.0f);
                dBeta.fill(0.0f);

                Tensor dInput = is_3d ? Tensor(batch_size, rows, self_d_model) : Tensor(rows, self_d_model);
                dInput.fill(0.0f);

                const float* output_grad_data = output_grad_cpu.raw();
                const float* gamma_data = gamma_cpu.raw();
                float* dGamma_data = dGamma.raw();
                float* dBeta_data = dBeta.raw();
                float* dInput_data = dInput.raw();

                if (is_3d) {
                    for (int b = 0; b < batch_size; b++) {
                        const int batch_offset = b * rows * self_d_model;
                        for (int i = 0; i < rows; i++) {
                            const int row_idx = b * rows + i;
                            const float std_inv = inv_stds[row_idx];
                            const float variance = (1.0f / (std_inv * std_inv)) - self_epsilon;
                            const float mean = means[row_idx];
                            const float* dout_row = output_grad_data + batch_offset + i * self_d_model;
                            const float* input_row = input_data + batch_offset + i * self_d_model;

                            float dvar = 0.0f;
                            for (int j = 0; j < self_d_model; j++) {
                                const float x_minus_mean = input_row[j] - mean;
                                const float normalized_ij = x_minus_mean * std_inv;
                                dGamma_data[j] += dout_row[j] * normalized_ij;
                                dBeta_data[j] += dout_row[j];
                                const float dnorm = dout_row[j] * gamma_data[j];
                                dvar += dnorm * x_minus_mean * -0.5f * std::pow(variance + self_epsilon, -1.5f);
                            }

                            float dmean = 0.0f;
                            for (int j = 0; j < self_d_model; j++) {
                                const float dnorm = dout_row[j] * gamma_data[j];
                                const float x_minus_mean = input_row[j] - mean;
                                dmean += dnorm * -std_inv + dvar * -2.0f * x_minus_mean / self_d_model;
                            }

                            float* dInput_row = dInput_data + batch_offset + i * self_d_model;
                            for (int j = 0; j < self_d_model; j++) {
                                const float dnorm = dout_row[j] * gamma_data[j];
                                const float x_minus_mean = input_row[j] - mean;
                                dInput_row[j] = dnorm * std_inv + dvar * 2.0f * x_minus_mean / self_d_model + dmean / self_d_model;
                            }
                        }
                    }
                } else {
                    for (int i = 0; i < rows; i++) {
                        const float std_inv = inv_stds[i];
                        const float variance = (1.0f / (std_inv * std_inv)) - self_epsilon;
                        const float mean = means[i];
                        const float* dout_row = output_grad_data + i * self_d_model;
                        const float* input_row = input_data + i * self_d_model;

                        float dvar = 0.0f;
                        for (int j = 0; j < self_d_model; j++) {
                            const float x_minus_mean = input_row[j] - mean;
                            const float normalized_ij = x_minus_mean * std_inv;
                            dGamma_data[j] += dout_row[j] * normalized_ij;
                            dBeta_data[j] += dout_row[j];
                            const float dnorm = dout_row[j] * gamma_data[j];
                            dvar += dnorm * x_minus_mean * -0.5f * std::pow(variance + self_epsilon, -1.5f);
                        }

                        float dmean = 0.0f;
                        for (int j = 0; j < self_d_model; j++) {
                            const float dnorm = dout_row[j] * gamma_data[j];
                            const float x_minus_mean = input_row[j] - mean;
                            dmean += dnorm * -std_inv + dvar * -2.0f * x_minus_mean / self_d_model;
                        }

                        float* dInput_row = dInput_data + i * self_d_model;
                        for (int j = 0; j < self_d_model; j++) {
                            const float dnorm = dout_row[j] * gamma_data[j];
                            const float x_minus_mean = input_row[j] - mean;
                            dInput_row[j] = dnorm * std_inv + dvar * 2.0f * x_minus_mean / self_d_model + dmean / self_d_model;
                        }
                    }
                }

                if (self_gamma->getData().getDevice() == Device::CUDA) {
                    self_gamma->getGrad().add_inplace(dGamma.to(Device::CUDA));
                } else {
                    self_gamma->getGrad().add_inplace(dGamma);
                }

                if (self_beta->getData().getDevice() == Device::CUDA) {
                    self_beta->getGrad().add_inplace(dBeta.to(Device::CUDA));
                } else {
                    self_beta->getGrad().add_inplace(dBeta);
                }

                if (self_input->getData().getDevice() == Device::CUDA) {
                    self_input->getGrad().add_inplace(dInput.to(Device::CUDA));
                } else {
                    self_input->getGrad().add_inplace(dInput);
                }
            });
        }
        return output;
    }

    if (!input_tensor.getIs3D()) {
        //2D case
        int rows = input_tensor.getRows();
        result = Tensor(rows, d_model, Device::CPU);

        const float* input_data = input_tensor.raw();
        const float* gamma_data = gamma->getData().raw();
        const float* beta_data = beta->getData().raw();
        float* result_data = result.raw();

        means.resize(rows);
        inv_stds.resize(rows);

        for (int i = 0; i < rows; i++) {
            const float* row_in = input_data + i * d_model;
            float* row_out = result_data + i * d_model;

            float mean = 0.0f;
            for (int j = 0; j < d_model; j++) {
                mean += row_in[j];
            }
            mean /= d_model;
            means[i] = mean;

            float variance = 0.0f;
            for (int j = 0; j < d_model; j++) {
                float diff = row_in[j] - mean;
                variance += diff * diff;
            }
            variance /= d_model;

            const float std_inv = 1.0f / std::sqrt(variance + epsilon);
            inv_stds[i] = std_inv;

            for (int j = 0; j < d_model; j++) {
                float norm = (row_in[j] - mean) * std_inv;
                row_out[j] = gamma_data[j] * norm + beta_data[j];
            }
        }

        auto output = Variable::create(result, input->requiresGrad());

        if (input->requiresGrad()) {
            auto self_input = input;
            auto self_gamma = gamma;
            auto self_beta = beta;
            int self_d_model = d_model;
            float self_epsilon = epsilon;

            output->addChild(input);
            output->addChild(gamma);
            output->addChild(beta);

            output->setBackwardFn([self_input, self_gamma, self_beta, output_weak = std::weak_ptr<Variable>(output), means, inv_stds, self_d_model, self_epsilon, rows]() {
                auto output = output_weak.lock();
                if (!output) return;

                Tensor dGamma(1, self_d_model);
                Tensor dBeta(1, self_d_model);
                dGamma.fill(0.0f);
                dBeta.fill(0.0f);

                Tensor dInput(rows, self_d_model);
                dInput.fill(0.0f);

                const float* output_grad_data = output->getGrad().raw();
                const float* gamma_data = self_gamma->getData().raw();
                const float* input_data = self_input->getData().raw();
                float* dGamma_data = dGamma.raw();
                float* dBeta_data = dBeta.raw();
                float* dInput_data = dInput.raw();

                for (int i = 0; i < rows; i++) {
                    const float std_inv = inv_stds[i];
                    const float variance = (1.0f / (std_inv * std_inv)) - self_epsilon;
                    const float mean = means[i];
                    const float* dout_row = output_grad_data + i * self_d_model;
                    const float* input_row = input_data + i * self_d_model;

                    float dvar = 0.0f;
                    for (int j = 0; j < self_d_model; j++) {
                        const float x_minus_mean = input_row[j] - mean;
                        const float normalized_ij = x_minus_mean * std_inv;

                        dGamma_data[j] += dout_row[j] * normalized_ij;
                        dBeta_data[j] += dout_row[j];

                        const float dnorm = dout_row[j] * gamma_data[j];
                        dvar += dnorm * x_minus_mean * -0.5f * std::pow(variance + self_epsilon, -1.5f);
                    }

                    float dmean = 0.0f;
                    for (int j = 0; j < self_d_model; j++) {
                        const float dnorm = dout_row[j] * gamma_data[j];
                        const float x_minus_mean = input_row[j] - mean;
                        dmean += dnorm * -std_inv + dvar * -2.0f * x_minus_mean / self_d_model;
                    }

                    float* dInput_row = dInput_data + i * self_d_model;
                    for (int j = 0; j < self_d_model; j++) {
                        const float dnorm = dout_row[j] * gamma_data[j];
                        const float x_minus_mean = input_row[j] - mean;
                        dInput_row[j] = dnorm * std_inv + dvar * 2.0f * x_minus_mean / self_d_model + dmean / self_d_model;
                    }
                }

                if (self_gamma->getData().getDevice() == Device::CUDA) {
                    self_gamma->getGrad().add_inplace(dGamma.to(Device::CUDA));
                } else {
                    self_gamma->getGrad().add_inplace(dGamma);
                }

                if (self_beta->getData().getDevice() == Device::CUDA) {
                    self_beta->getGrad().add_inplace(dBeta.to(Device::CUDA));
                } else {
                    self_beta->getGrad().add_inplace(dBeta);
                }

                if (self_input->getData().getDevice() == Device::CUDA) {
                    self_input->getGrad().add_inplace(dInput.to(Device::CUDA));
                } else {
                    self_input->getGrad().add_inplace(dInput);
                }
            });
        }

        return output;

    } else {
        // 3D case
        int batch_size = input_tensor.getBatchSize();
        int seq_len = input_tensor.getRows();

        result = Tensor(batch_size, seq_len, d_model, Device::CPU);

        const float* input_data = input_tensor.raw();
        const float* gamma_data = gamma->getData().raw();
        const float* beta_data = beta->getData().raw();
        float* result_data = result.raw();

        int total_rows = batch_size * seq_len;
        means.resize(total_rows);
        inv_stds.resize(total_rows);

        for (int b = 0; b < batch_size; b++) {
            const int batch_offset = b * seq_len * d_model;
            
            for (int i = 0; i < seq_len; i++) {
                const int row_idx = b * seq_len + i;
                const float* row_in = input_data + batch_offset + i * d_model;
                float* row_out = result_data + batch_offset + i * d_model;
  
                float mean = 0.0f;
                for (int j = 0; j < d_model; j++) {
                    mean += row_in[j];
                }
                mean /= d_model;
                means[row_idx] = mean;

                float variance = 0.0f;
                for (int j = 0; j < d_model; j++) {
                    float diff = row_in[j] - mean;
                    variance += diff * diff;
                }
                variance /= d_model;

                const float std_inv = 1.0f / std::sqrt(variance + epsilon);
                inv_stds[row_idx] = std_inv;

                for (int j = 0; j < d_model; j++) {
                    float norm = (row_in[j] - mean) * std_inv;
                    row_out[j] = gamma_data[j] * norm + beta_data[j];
                }
            }
        }

        auto output = Variable::create(result, input->requiresGrad());

        if (input->requiresGrad()) {
            auto self_input = input;
            auto self_gamma = gamma;
            auto self_beta = beta;
            int self_d_model = d_model;
            float self_epsilon = epsilon;

            output->addChild(input);
            output->addChild(gamma);
            output->addChild(beta);

            output->setBackwardFn([self_input, self_gamma, self_beta, output_weak = std::weak_ptr<Variable>(output), means, inv_stds, self_d_model, self_epsilon, batch_size, seq_len]() {
                auto output = output_weak.lock();
                if (!output) return;

                Tensor dGamma(1, self_d_model);
                Tensor dBeta(1, self_d_model);
                dGamma.fill(0.0f);
                dBeta.fill(0.0f);

                Tensor dInput(batch_size, seq_len, self_d_model);
                dInput.fill(0.0f);

                const float* output_grad_data = output->getGrad().raw();
                const float* gamma_data = self_gamma->getData().raw();
                const float* input_data = self_input->getData().raw();
                float* dGamma_data = dGamma.raw();
                float* dBeta_data = dBeta.raw();
                float* dInput_data = dInput.raw();

                for (int b = 0; b < batch_size; b++) {
                    const int batch_offset = b * seq_len * self_d_model;

                    for (int i = 0; i < seq_len; i++) {
                        const int row_idx = b * seq_len + i;
                        const float std_inv = inv_stds[row_idx];
                        const float variance = (1.0f / (std_inv * std_inv)) - self_epsilon;
                        const float mean = means[row_idx];

                        const float* dout_row = output_grad_data + batch_offset + i * self_d_model;
                        const float* input_row = input_data + batch_offset + i * self_d_model;

                        float dvar = 0.0f;
                        for (int j = 0; j < self_d_model; j++) {
                            const float x_minus_mean = input_row[j] - mean;
                            const float normalized_ij = x_minus_mean * std_inv;

                            dGamma_data[j] += dout_row[j] * normalized_ij;
                            dBeta_data[j] += dout_row[j];

                            const float dnorm = dout_row[j] * gamma_data[j];
                            dvar += dnorm * x_minus_mean * -0.5f * std::pow(variance + self_epsilon, -1.5f);
                        }

                        float dmean = 0.0f;
                        for (int j = 0; j < self_d_model; j++) {
                            const float dnorm = dout_row[j] * gamma_data[j];
                            const float x_minus_mean = input_row[j] - mean;
                            dmean += dnorm * -std_inv + dvar * -2.0f * x_minus_mean / self_d_model;
                        }

                        float* dInput_row = dInput_data + batch_offset + i * self_d_model;
                        for (int j = 0; j < self_d_model; j++) {
                            const float dnorm = dout_row[j] * gamma_data[j];
                            const float x_minus_mean = input_row[j] - mean;
                            dInput_row[j] = dnorm * std_inv + dvar * 2.0f * x_minus_mean / self_d_model + dmean / self_d_model;
                        }
                    }
                }

                if (self_gamma->getData().getDevice() == Device::CUDA) {
                    self_gamma->getGrad().add_inplace(dGamma.to(Device::CUDA));
                } else {
                    self_gamma->getGrad().add_inplace(dGamma);
                }

                if (self_beta->getData().getDevice() == Device::CUDA) {
                    self_beta->getGrad().add_inplace(dBeta.to(Device::CUDA));
                } else {
                    self_beta->getGrad().add_inplace(dBeta);
                }

                if (self_input->getData().getDevice() == Device::CUDA) {
                    self_input->getGrad().add_inplace(dInput.to(Device::CUDA));
                } else {
                    self_input->getGrad().add_inplace(dInput);
                }
            });
        }
        return output;
    }
}