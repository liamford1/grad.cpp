#pragma once

#include "tensor.h"
#include "variable.h"
#include <memory>

class MultiHeadAttention {
    private:
        int d_model;
        int num_heads;
        float dropout_rate;

        std::shared_ptr<Variable> W_q;
        std::shared_ptr<Variable> W_k;
        std::shared_ptr<Variable> W_v;
        std::shared_ptr<Variable> W_o;
    public:
        MultiHeadAttention(int d_model, int num_heads, float dropout_rate = 0.1f);
        ~MultiHeadAttention();
        std::shared_ptr<Variable> forward(std::shared_ptr<Variable> input, bool training = false) const;

        const std::shared_ptr<Variable> getW_q() const { return W_q; }
        const std::shared_ptr<Variable> getW_k() const { return W_k; }
        const std::shared_ptr<Variable> getW_v() const { return W_v; }
        const std::shared_ptr<Variable> getW_o() const { return W_o; }

        std::vector<std::shared_ptr<Variable>> parameters() const {
            return {W_q, W_k, W_v, W_o};
        }

        void setWeights(const Tensor& wq, const Tensor& wk, const Tensor& wv, const Tensor& wo) {
            W_q = Variable::create(wq, true);
            W_k = Variable::create(wk, true);
            W_v = Variable::create(wv, true);
            W_o = Variable::create(wo, true);
        }
};
