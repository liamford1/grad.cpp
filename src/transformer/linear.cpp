#include "transformer/linear.h"
#include <iostream>

Linear::Linear(int input_dim, int output_dim, bool use_bias) :
    input_dim(input_dim),
    output_dim(output_dim),
    use_bias(use_bias)
{
    Tensor w_tensor(input_dim, output_dim);
    w_tensor.xavier(input_dim, output_dim);
    weights = Variable::create(w_tensor, true);
    registerParameter(weights);

    if (use_bias) { 
        Tensor b_tensor(1, output_dim);
        b_tensor.fill(0.0f);
        bias = Variable::create(b_tensor, true);
        registerParameter(bias);
    }
}

std::shared_ptr<Variable> Linear::forward(std::shared_ptr<Variable> input) const {
    auto result = input->matmul(weights);
    if (use_bias && bias) {
        result = result->add(bias);
    }
    return result;
}