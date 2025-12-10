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

    if (!input_tensor.getIs3D()) {
        //2D case
        int rows = input_tensor.getRows();
        Tensor result(rows, d_model);
        
        const float* input_data = input_tensor.raw();
        const float* gamma_data = gamma->getData().raw();
        const float* beta_data = beta->getData().raw();
        float* result_data = result.raw();
        
        std::vector<float> means(rows);
        std::vector<float> inv_stds(rows);

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

            output->setBackwardFn([self_input, self_gamma, self_beta, output, means, inv_stds, self_d_model, self_epsilon, rows]() {
                
                Tensor dGamma(1, self_d_model);
                Tensor dBeta(1, self_d_model);
                dGamma.fill(0.0f);
                dBeta.fill(0.0f);

                Tensor dInput(rows, self_d_model);
                dInput.fill(0.0f);

                for (int i = 0; i < rows; i++) {
                    float std_inv = inv_stds[i];
                    float variance = (1.0f / (std_inv * std_inv)) - self_epsilon;

                    float dvar = 0.0f;
                    for (int j = 0; j < self_d_model; j++) {
                        float dout = output->getGrad().getValue(i, j);
                        float gamma_val = self_gamma->getData().getValue(0, j);

                        float x_minus_mean = self_input->getData().getValue(i, j) - means[i];
                        float normalized_ij = x_minus_mean * std_inv;

                        dGamma.setValue(0, j, dGamma.getValue(0, j) + dout * normalized_ij);
                        dBeta.setValue(0, j, dBeta.getValue(0, j) + dout);

                        float dnorm = dout * gamma_val;

                        dvar += dnorm * x_minus_mean * -0.5f * std::pow(variance + self_epsilon, -1.5f);
                    }

                    float dmean = 0.0f;
                    for (int j = 0; j < self_d_model; j++) {
                        float dout = output->getGrad().getValue(i, j);
                        float gamma_val = self_gamma->getData().getValue(0, j);
                        float dnorm = dout * gamma_val;
                        float x_minus_mean = self_input->getData().getValue(i, j) - means[i];

                        dmean += dnorm * -std_inv + dvar * -2.0f * x_minus_mean / self_d_model;
                    }

                    for (int j = 0; j < self_d_model; j++) {
                        float dout = output->getGrad().getValue(i, j);
                        float gamma_val = self_gamma->getData().getValue(0, j);
                        float dnorm = dout * gamma_val;
                        float x_minus_mean = self_input->getData().getValue(i, j) - means[i];

                        float dx = dnorm * std_inv + dvar * 2.0f * x_minus_mean / self_d_model + dmean / self_d_model;
                        dInput.setValue(i, j, dx);
                    }
                }

                self_gamma->getGrad().add_inplace(dGamma);
                self_beta->getGrad().add_inplace(dBeta);
                self_input->getGrad().add_inplace(dInput);
            });
        }

        return output;

    } else {
        // 3D case
        int batch_size = input_tensor.getBatchSize();
        int seq_len = input_tensor.getRows();
        
        Tensor result(batch_size, seq_len, d_model);
        
        const float* input_data = input_tensor.raw();
        const float* gamma_data = gamma->getData().raw();
        const float* beta_data = beta->getData().raw();
        float* result_data = result.raw();
        
        int total_rows = batch_size * seq_len;
        std::vector<float> means(total_rows);
        std::vector<float> inv_stds(total_rows);

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

            output->setBackwardFn([self_input, self_gamma, self_beta, output, means, inv_stds, self_d_model, self_epsilon, batch_size, seq_len]() {
                
                Tensor dGamma(1, self_d_model);
                Tensor dBeta(1, self_d_model);
                dGamma.fill(0.0f);
                dBeta.fill(0.0f);

                Tensor dInput(batch_size, seq_len, self_d_model);
                dInput.fill(0.0f);

                for (int b = 0; b < batch_size; b++) {
                    for (int i = 0; i < seq_len; i++) {
                        const int row_idx = b * seq_len + i;
                        
                        float std_inv = inv_stds[row_idx];
                        float variance = (1.0f / (std_inv * std_inv)) - self_epsilon;

                        float dvar = 0.0f;
                        for (int j = 0; j < self_d_model; j++) {
                            float dout = output->getGrad().getValue(b, i, j);
                            float gamma_val = self_gamma->getData().getValue(0, j);

                            float x_minus_mean = self_input->getData().getValue(b, i, j) - means[row_idx];
                            float normalized_ij = x_minus_mean * std_inv;

                            dGamma.setValue(0, j, dGamma.getValue(0, j) + dout * normalized_ij);
                            dBeta.setValue(0, j, dBeta.getValue(0, j) + dout);

                            float dnorm = dout * gamma_val;

                            dvar += dnorm * x_minus_mean * -0.5f * std::pow(variance + self_epsilon, -1.5f);
                        }

                        float dmean = 0.0f;
                        for (int j = 0; j < self_d_model; j++) {
                            float dout = output->getGrad().getValue(b, i, j);
                            float gamma_val = self_gamma->getData().getValue(0, j);
                            float dnorm = dout * gamma_val;
                            float x_minus_mean = self_input->getData().getValue(b, i, j) - means[row_idx];

                            dmean += dnorm * -std_inv + dvar * -2.0f * x_minus_mean / self_d_model;
                        }

                        for (int j = 0; j < self_d_model; j++) {
                            float dout = output->getGrad().getValue(b, i, j);
                            float gamma_val = self_gamma->getData().getValue(0, j);
                            float dnorm = dout * gamma_val;
                            float x_minus_mean = self_input->getData().getValue(b, i, j) - means[row_idx];

                            float dx = dnorm * std_inv + dvar * 2.0f * x_minus_mean / self_d_model + dmean / self_d_model;
                            dInput.setValue(b, i, j, dx);
                        }
                    }
                }

                self_gamma->getGrad().add_inplace(dGamma);
                self_beta->getGrad().add_inplace(dBeta);
                self_input->getGrad().add_inplace(dInput);
            });
        }
        return output;
    }
}