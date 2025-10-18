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
        int rows = input_tensor.getRows();
        Tensor result(rows, d_model);
        
        std::vector<float> means(rows);
        std::vector<float> variances(rows);
        std::vector<std::vector<float>> normalized(rows, std::vector<float>(d_model));

        for (int i = 0; i < rows; i++) {
            float mean = 0.0f;
            for (int j = 0; j < d_model; j++) {
                mean += input_tensor.getValue(i, j);
            }
            mean = mean / d_model;
            means[i] = mean;

            float variance = 0.0f;
            for (int j = 0; j < d_model; j++) {
                float diff = input_tensor.getValue(i, j) - mean;
                variance += diff * diff;
            }
            variance = variance / d_model;
            variances[i] = variance;

            for (int j = 0; j < d_model; j++) {
                float norm = (input_tensor.getValue(i, j) - mean) / std::sqrt(variance + epsilon);
                normalized[i][j] = norm;
                float output = gamma->getData().getValue(0, j) * norm + beta->getData().getValue(0, j);
                result.setValue(i, j, output);
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

            output->setBackwardFn([self_input, self_gamma, self_beta, output, means, variances, 
                                   normalized, self_d_model, self_epsilon, rows]() {
                
                Tensor dGamma(1, self_d_model);
                Tensor dBeta(1, self_d_model);
                dGamma.fill(0.0f);
                dBeta.fill(0.0f);

                Tensor dInput(rows, self_d_model);
                dInput.fill(0.0f);

                for (int i = 0; i < rows; i++) {
                    float variance = variances[i];
                    float std_inv = 1.0f / std::sqrt(variance + self_epsilon);

                    float dvar = 0.0f;
                    for (int j = 0; j < self_d_model; j++) {
                        float dout = output->getGrad().getValue(i, j);
                        float gamma_val = self_gamma->getData().getValue(0, j);

                        dGamma.setValue(0, j, dGamma.getValue(0, j) + dout * normalized[i][j]);
                        dBeta.setValue(0, j, dBeta.getValue(0, j) + dout);

                        float dnorm = dout * gamma_val;
                        float x_minus_mean = self_input->getData().getValue(i, j) - means[i];

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
        int batch_size = input_tensor.getBatchSize();
        int seq_len = input_tensor.getRows();
        
        Tensor result(batch_size, seq_len, d_model);
        
        std::vector<std::vector<float>> means(batch_size, std::vector<float>(seq_len));
        std::vector<std::vector<float>> variances(batch_size, std::vector<float>(seq_len));
        std::vector<std::vector<std::vector<float>>> normalized(batch_size, 
            std::vector<std::vector<float>>(seq_len, std::vector<float>(d_model)));

        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < seq_len; i++) {
                float mean = 0.0f;
                for (int j = 0; j < d_model; j++) {
                    mean += input_tensor.getValue(b, i, j);
                }
                mean = mean / d_model;
                means[b][i] = mean;

                float variance = 0.0f;
                for (int j = 0; j < d_model; j++) {
                    float diff = input_tensor.getValue(b, i, j) - mean;
                    variance += diff * diff;
                }
                variance = variance / d_model;
                variances[b][i] = variance;

                for (int j = 0; j < d_model; j++) {
                    float norm = (input_tensor.getValue(b, i, j) - mean) / std::sqrt(variance + epsilon);
                    normalized[b][i][j] = norm;
                    float output = gamma->getData().getValue(0, j) * norm + beta->getData().getValue(0, j);
                    result.setValue(b, i, j, output);
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

            output->setBackwardFn([self_input, self_gamma, self_beta, output, means, variances, 
                                   normalized, self_d_model, self_epsilon, batch_size, seq_len]() {
                
                Tensor dGamma(1, self_d_model);
                Tensor dBeta(1, self_d_model);
                dGamma.fill(0.0f);
                dBeta.fill(0.0f);

                Tensor dInput(batch_size, seq_len, self_d_model);
                dInput.fill(0.0f);

                for (int b = 0; b < batch_size; b++) {
                    for (int i = 0; i < seq_len; i++) {
                        float variance = variances[b][i];
                        float std_inv = 1.0f / std::sqrt(variance + self_epsilon);

                        // Step 1: Compute dvar
                        float dvar = 0.0f;
                        for (int j = 0; j < self_d_model; j++) {
                            float dout = output->getGrad().getValue(b, i, j);
                            float gamma_val = self_gamma->getData().getValue(0, j);

                            dGamma.setValue(0, j, dGamma.getValue(0, j) + dout * normalized[b][i][j]);
                            dBeta.setValue(0, j, dBeta.getValue(0, j) + dout);

                            float dnorm = dout * gamma_val;
                            float x_minus_mean = self_input->getData().getValue(b, i, j) - means[b][i];

                            dvar += dnorm * x_minus_mean * -0.5f * std::pow(variance + self_epsilon, -1.5f);
                        }

                        // Step 2: Compute dmean (after dvar is complete)
                        float dmean = 0.0f;
                        for (int j = 0; j < self_d_model; j++) {
                            float dout = output->getGrad().getValue(b, i, j);
                            float gamma_val = self_gamma->getData().getValue(0, j);
                            float dnorm = dout * gamma_val;
                            float x_minus_mean = self_input->getData().getValue(b, i, j) - means[b][i];

                            dmean += dnorm * -std_inv + dvar * -2.0f * x_minus_mean / self_d_model;
                        }

                        // Step 3: Compute dx
                        for (int j = 0; j < self_d_model; j++) {
                            float dout = output->getGrad().getValue(b, i, j);
                            float gamma_val = self_gamma->getData().getValue(0, j);
                            float dnorm = dout * gamma_val;
                            float x_minus_mean = self_input->getData().getValue(b, i, j) - means[b][i];

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