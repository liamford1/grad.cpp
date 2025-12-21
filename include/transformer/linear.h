#pragma once
#include "module.h"
#include "tensor.h"
#include "variable.h"
#include <memory>

class Linear : public Module {
    private:
        std::shared_ptr<Variable> weights;
        std::shared_ptr<Variable> bias;
        bool use_bias;
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