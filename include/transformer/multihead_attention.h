#ifndef MULTIHEAD_ATTENTION_H
#define MULTIHEAD_ATTENTION_H

#include "tensor.h"

class MultiHeadAttention {
    private:
        int d_model;
        int num_heads;
        float dropout_rate;

        Tensor W_q;
        Tensor W_k;
        Tensor W_v;
        Tensor W_o;
    public:
        MultiHeadAttention(int d_model, int num_heads, float dropout_rate = 0.1f);
        ~MultiHeadAttention();
        Tensor forward(const Tensor& input, bool training = false) const;

        const Tensor& getW_q() const { return W_q; }
        const Tensor& getW_k() const { return W_k; }
        const Tensor& getW_v() const { return W_v; }
        const Tensor& getW_o() const { return W_o; }

        void setWeights(const Tensor& wq, const Tensor& wk, const Tensor& wv, const Tensor& wo) {
            W_q = wq;
            W_k = wk;
            W_v = wv;
            W_o = wo;
        }
};

#endif