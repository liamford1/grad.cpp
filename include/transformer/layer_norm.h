#pragma once

#include "tensor.h"
#include "variable.h"
#include <memory>

class LayerNorm {
    private:
        int d_model;
        Tensor gamma;
        Tensor beta;
        float epsilon;
    public:
        LayerNorm(int d_model);
        ~LayerNorm();
        std::shared_ptr<Variable> forward(std::shared_ptr<Variable> input) const;

        const Tensor& getGamma() const { return gamma; }
        const Tensor& getBeta() const { return beta; }

        void setParams(const Tensor& new_gamma, const Tensor& new_beta) {
            gamma = new_gamma;
            beta = new_beta;
        }
};