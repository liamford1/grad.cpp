#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include "tensor.h"
#include "multihead_attention.h"
#include "layer_norm.h"
#include "feedforward.h"

class TransformerBlock {
    private:
        MultiHeadAttention attention;
        FeedForward ffn;
        LayerNorm norm1;
        LayerNorm norm2;
        float dropout_rate;
    public:
        TransformerBlock(int d_model, int num_heads, int ffn_hidden_dim = -1, float dropout_rate = 0.1f);
        Tensor forward(const Tensor& input, bool training = false) const;
};

#endif