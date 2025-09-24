#include "tensor.h"
#include "layer_norm.h"
#include <iostream>
#include <cmath>

LayerNorm::LayerNorm(int d_model) : 
    d_model(d_model),
    gamma(1, d_model),
    beta(1, d_model)
{
    this->epsilon = 1e-5f;

    gamma.fill(1.0f);
    beta.fill(0.0f);
}

LayerNorm::~LayerNorm() {}

Tensor LayerNorm::forward(const Tensor& input) const {
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
}