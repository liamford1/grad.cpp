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

Tensor LayerNorm::forward(const Tensor& input) const {
    if (!input.getIs3D()) {
        Tensor result(input.getRows(), input.getCols());

        for (int i = 0; i < input.getRows(); i++) {
            float mean = 0.0f;
            float variance = 0.0f;

            for (int j = 0; j < d_model; j++) {
                mean += input.getValue(i, j);
            }
            mean = mean / d_model;

            for (int j = 0; j < d_model; j++) {
                float diff = input.getValue(i, j) - mean;
                variance += diff * diff;
            }
            variance = variance / d_model;

            for (int j = 0; j < d_model; j++) {
                float normalized = (input.getValue(i, j) - mean) / std::sqrt(variance + epsilon);
                float output = gamma.getValue(0, j) * normalized + beta.getValue(0, j);
                result.setValue(i, j, output);
            }
        }
        return result;
    } else {
        int batch_size = input.getBatchSize();
        int seq_len = input.getRows();
        
        Tensor result(batch_size, seq_len, d_model);

        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < seq_len; i++) {
                float mean = 0.0f;
                float variance = 0.0f;

                for (int j = 0; j < d_model; j++) {
                    mean += input.getValue(b, i, j);
                }
                mean = mean / d_model;

                for (int j = 0; j < d_model; j++) {
                    float diff = input.getValue(b, i, j) - mean;
                    variance += diff * diff;
                }
                variance = variance / d_model;

                for (int j = 0; j < d_model; j++) {
                    float normalized = (input.getValue(b, i, j) - mean) / std::sqrt(variance + epsilon);
                    float output = gamma.getValue(0, j) * normalized + beta.getValue(0, j);
                    result.setValue(b, i, j, output);
                }
            }
        }
        return result;
    }
}