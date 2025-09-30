#pragma once
#include "tensor.h"
#include "variable.h"
#include <memory>

class LayerNorm {
    private:
        int d_model;
        std::shared_ptr<Variable> gamma;
        std::shared_ptr<Variable> beta;
        float epsilon;
    public:
        LayerNorm(int d_model);
        ~LayerNorm();
        std::shared_ptr<Variable> forward(std::shared_ptr<Variable> input) const;

        std::shared_ptr<Variable> getGamma() const { return gamma; }
        std::shared_ptr<Variable> getBeta() const { return beta; }

        std::vector<std::shared_ptr<Variable>> parameters() const {
            return {gamma, beta};
        }

        void setParams(const Tensor& new_gamma, const Tensor& new_beta) {
            gamma = Variable::create(new_gamma, true);
            beta = Variable::create(new_beta, true);
        }
};