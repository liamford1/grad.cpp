#include "transformer/tensor.h"
#include "transformer/attention.h"
#include <iostream>

Attention::Attention(int d_model, int d_k, int d_v) :
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

Tensor Attention::forward(const Tensor& input) const {
    Tensor Q = input.matmul(W_q);
    Tensor K = input.matmul(W_k);
    Tensor V = input.matmul(W_v);

    Tensor scores = Q.matmul(K.transpose());

    Tensor attention_weights = scores.softmax();

    Tensor attended_values = attention_weights.matmul(V);

    Tensor output = attended_values.matmul(W_o);

    return output;
}