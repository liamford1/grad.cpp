#ifndef ATTENTION_H
#define ATTENTION_H

#include "tensor.h"

class Attention {
    private:
        int d_model;
        int d_k;
        int d_v;

        Tensor W_q;
        Tensor W_k;
        Tensor W_v;
        Tensor W_o;
    public:
        Attention(int d_model, int d_k, int d_v);

        ~Attention();

        Tensor forward(const Tensor& input) const;
};

#endif