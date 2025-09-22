#include "tensor.h"
#include "attention.h"
#include <iostream>

Attention::Attention(int d_model, int d_k, int d_v) : 
    d_model(d_model), d_k(d_k), d_v(d_v),
    W_q(d_model, d_k),
    W_k(d_model, d_k), 
    W_v(d_model, d_v),
    W_o(d_v, d_model) {

    W_q.fill(0.1);
    W_k.fill(0.2);
    W_v.fill(0.3);
    W_o.fill(0.4);
}

Attention::~Attention() {}