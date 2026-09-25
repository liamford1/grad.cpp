#include "transformer/linear.h"

Linear::Linear(int input_dim, int output_dim, bool use_bias) :
    use_bias(use_bias)
{
    Tensor w_tensor(input_dim, output_dim);
    w_tensor.xavier(input_dim, output_dim);
    weights = Variable::create(w_tensor, true);

    if (use_bias) {
        bias = Variable::create(Tensor(1, output_dim), true);  // zero-initialized
    }
}

std::shared_ptr<Variable> Linear::forward(std::shared_ptr<Variable> input) const {
    auto result = input->matmul(weights);
    if (use_bias && bias) {
        result = result->add(bias);
    }
    return result;
}