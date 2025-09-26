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
};

#endif