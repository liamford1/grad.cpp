#include "transformer/tensor.h"
#include "transformer/multihead_attention.h"
#include "transformer/layer_norm.h"
#include "transformer/feedforward.h"
#include "transformer/transformer_block.h"
#include <iostream>

TransformerBlock::TransformerBlock(int d_model, int num_heads, int ffn_hidden_dim) :
    attention(d_model, num_heads),
    ffn(d_model, ffn_hidden_dim),
    norm1(d_model),
    norm2(d_model) {}

Tensor TransformerBlock::forward(const Tensor& input) const {
    Tensor normed1 = norm1.forward(input);
    Tensor attention_output = attention.forward(normed1);
    Tensor residual = input.add(attention_output);

    Tensor normed2 = norm2.forward(residual);
    Tensor ffn_output = ffn.forward(normed2);
    return residual.add(ffn_output);
}