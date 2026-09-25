#pragma once
#include "grad/transformer/tensor.h"
#include "grad/transformer/variable.h"
#include "grad/transformer/linear.h"
#include <memory>

class MultiHeadAttention {
    private:
        int d_model;
        int num_heads;
        float dropout_rate;
        // Rotary position embeddings: Q and K are rotated by position-
        // dependent angles before the score matmul, encoding relative
        // position directly in the attention instead of via learned
        // absolute embeddings added to the residual stream.
        bool rope_;

        std::shared_ptr<Variable> W_q;
        std::shared_ptr<Variable> W_k;
        std::shared_ptr<Variable> W_v;
        std::shared_ptr<Variable> W_o;
        std::shared_ptr<Variable> b_q;
        std::shared_ptr<Variable> b_k;
        std::shared_ptr<Variable> b_v;
        std::shared_ptr<Variable> b_o;
    public:
        MultiHeadAttention(int d_model, int num_heads, float dropout_rate = 0.1f,
                           bool rope = false);
        bool usesRope() const { return rope_; }
        std::shared_ptr<Variable> forward(std::shared_ptr<Variable> input, bool training = false) const;

        const std::shared_ptr<Variable>& getW_q() const { return W_q; }
        const std::shared_ptr<Variable>& getW_k() const { return W_k; }
        const std::shared_ptr<Variable>& getW_v() const { return W_v; }
        const std::shared_ptr<Variable>& getW_o() const { return W_o; }
        const std::shared_ptr<Variable>& getB_q() const { return b_q; }
        const std::shared_ptr<Variable>& getB_k() const { return b_k; }
        const std::shared_ptr<Variable>& getB_v() const { return b_v; }
        const std::shared_ptr<Variable>& getB_o() const { return b_o; }

        std::vector<std::shared_ptr<Variable>> parameters() const {
            return {W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o};
        }
};
