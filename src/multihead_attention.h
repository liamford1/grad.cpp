#ifndef MULTIHEAD_ATTENTION_H
#define MULTIHEAD_ATTENTION_H

#include "tensor.h"

class MultiHeadAttention {
    private:
        int d_model;
        int num_heads;

        Tensor W_q;
        Tensor W_k;
        Tensor W_v;
        Tensor W_o;
    public:
        MultiHeadAttention(int d_model, int num_heads);

        ~MultiHeadAttention();

        Tensor forward(const Tensor& input) const;
};

#endif