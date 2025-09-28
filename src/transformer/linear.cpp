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
        if (input.getIs3D()) {
            int seq_len = input.getRows();
            Tensor expanded_bias(seq_len, output_dim);
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < output_dim; j++) {
                    expanded_bias.setValue(i, j, bias.getValue(0, j));
                }
            }
            result = result.add(expanded_bias);
        } else {
            result = result.add(bias);
        }
    }
    return result;
}