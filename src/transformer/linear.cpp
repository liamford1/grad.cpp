#include "grad/transformer/linear.h"
#include "grad/utils/narrow.h"

namespace grad {

Linear::Linear(int input_dim, int output_dim, bool use_bias) :
    use_bias_(use_bias)
{
    const size_t in = narrow<size_t>(input_dim);
    const size_t out = narrow<size_t>(output_dim);
    Tensor w_tensor(in, out);
    w_tensor.xavier(in, out);
    weights = Variable::create(w_tensor, true);

    if (use_bias) {
        bias = Variable::create(Tensor(1, out), true);  // zero-initialized
    }
}

std::shared_ptr<Variable> Linear::forward(std::shared_ptr<Variable> input) const {
    auto result = input->matmul(weights);
    if (use_bias_ && bias) {
        result = result->add(bias);
    }
    return result;
}

}  // namespace grad
