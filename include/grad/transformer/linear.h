#pragma once
#include "grad/transformer/tensor.h"
#include "grad/transformer/variable.h"
#include <memory>

namespace grad {

class Linear {
private:
    std::shared_ptr<Variable> weights;
    std::shared_ptr<Variable> bias;
    bool use_bias_;

public:
    Linear(int input_dim, int output_dim, bool use_bias = true);

    std::shared_ptr<Variable> forward(std::shared_ptr<Variable> input) const;

    std::shared_ptr<Variable> getWeights() const { return weights; }
    std::shared_ptr<Variable> getBias() const { return bias; }

    void setWeights(std::shared_ptr<Variable> new_weights, std::shared_ptr<Variable> new_bias) {
        weights = new_weights;
        bias = new_bias;
    }
};

}  // namespace grad
