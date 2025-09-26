#include "transformer/tensor.h"
#include "transformer/linear.h"
#include "transformer/feedforward.h"
#include <iostream>
#include <cmath>

Tensor gelu(const Tensor& input) {
    Tensor result(input.getRows(), input.getCols());
    for (int i = 0; i < input.getRows(); i++) {
        for (int j = 0; j < input.getCols(); j++) {
            float x = input.getValue(i, j);
            float gelu_val = 0.5f * x * (1.0f + tanh(0.79788456f * (x + 0.044715f * x * x * x)));
            result.setValue(i, j, gelu_val);
        }
    }
    return result;
}

FeedForward::FeedForward(int d_model, int hidden_dim) :
    layer1(d_model, (hidden_dim == -1) ? 4 * d_model : hidden_dim),
    layer2((hidden_dim == -1) ? 4 * d_model : hidden_dim, d_model) {}

Tensor FeedForward::forward(const Tensor& input) const {
    Tensor output = layer1.forward(input);
    output = gelu(output);
    return layer2.forward(output);
}