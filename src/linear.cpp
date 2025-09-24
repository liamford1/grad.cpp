#include "transformer/tensor.h"
#include "transformer/linear.h"
#include <iostream>

Linear::Linear(int input_dim, int output_dim, bool use_bias) :
    input_dim(input_dim),
    output_dim(output_dim),
    weights(input_dim, output_dim),
    bias(1, output_dim),
    use_bias(use_bias)
{
    weights.xavier(input_dim, output_dim);
    if (use_bias) { bias.fill(0.0f); }
}

Linear::~Linear() {}

Tensor Linear::forward(const Tensor& input) const {
    Tensor result = input.matmul(weights);

    if (use_bias) {
        result = result.add(bias);
    }
    return result;
}