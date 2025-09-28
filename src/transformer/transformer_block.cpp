#include "transformer/tensor.h"
#include "transformer/multihead_attention.h"
#include "transformer/layer_norm.h"
#include "transformer/feedforward.h"
#include "transformer/activations.h"
#include "transformer/transformer_block.h"
#include <iostream>

TransformerBlock::TransformerBlock(int d_model, int num_heads, int ffn_hidden_dim, float dropout_rate) :
    attention(d_model, num_heads, dropout_rate),
    ffn(d_model, ffn_hidden_dim, dropout_rate),
    dropout_rate(dropout_rate),
    norm1(d_model),
    norm2(d_model) {}

std::shared_ptr<Variable> TransformerBlock::forward(std::shared_ptr<Variable> input, bool training) const {
    auto normed1 = norm1.forward(input);
    auto attention_output = attention.forward(normed1, training);
    attention_output = attention_output->dropout(dropout_rate, training);
    auto residual1 = input->add(attention_output);

    auto normed2 = norm2.forward(residual1);
    auto ffn_output = ffn.forward(normed2, training);
    ffn_output = ffn_output->dropout(dropout_rate, training);
    return residual1->add(ffn_output);
}