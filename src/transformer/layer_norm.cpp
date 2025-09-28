#include "transformer/tensor.h"
#include "transformer/layer_norm.h"
#include <iostream>
#include <cmath>

LayerNorm::LayerNorm(int d_model) : 
    d_model(d_model),
    gamma(1, d_model),
    beta(1, d_model) {
    this->epsilon = 1e-5f;
    gamma.fill(1.0f);
    beta.fill(0.0f);
}

LayerNorm::~LayerNorm() {}

std::shared_ptr<Variable> LayerNorm::forward(std::shared_ptr<Variable> input) const {
    Tensor input_tensor = input->getData();

    if (!input_tensor.getIs3D()) {
        Tensor result(input_tensor.getRows(), input_tensor.getCols());

        for (int i = 0; i < input_tensor.getRows(); i++) {
            float mean = 0.0f;
            float variance = 0.0f;

            for (int j = 0; j < d_model; j++) {
                mean += input_tensor.getValue(i, j);
            }
            mean = mean / d_model;

            for (int j = 0; j < d_model; j++) {
                float diff = input_tensor.getValue(i, j) - mean;
                variance += diff * diff;
            }
            variance = variance / d_model;

            for (int j = 0; j < d_model; j++) {
                float normalized = (input_tensor.getValue(i, j) - mean) / std::sqrt(variance + epsilon);
                float output = gamma.getValue(0, j) * normalized + beta.getValue(0, j);
                result.setValue(i, j, output);
            }
        }
        return Variable::create(result, input->requiresGrad());
    } else {
        int batch_size = input_tensor.getBatchSize();
        int seq_len = input_tensor.getRows();
        
        Tensor result(batch_size, seq_len, d_model);

        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < seq_len; i++) {
                float mean = 0.0f;
                float variance = 0.0f;

                for (int j = 0; j < d_model; j++) {
                    mean += input_tensor.getValue(b, i, j);
                }
                mean = mean / d_model;

                for (int j = 0; j < d_model; j++) {
                    float diff = input_tensor.getValue(b, i, j) - mean;
                    variance += diff * diff;
                }
                variance = variance / d_model;

                for (int j = 0; j < d_model; j++) {
                    float normalized = (input_tensor.getValue(b, i, j) - mean) / std::sqrt(variance + epsilon);
                    float output = gamma.getValue(0, j) * normalized + beta.getValue(0, j);
                    result.setValue(b, i, j, output);
                }
            }
        }
        return Variable::create(result, input->requiresGrad());
    }
}