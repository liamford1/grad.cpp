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
    norm1(d_model),
    norm2(d_model) {}

std::shared_ptr<Variable> TransformerBlock::forward(std::shared_ptr<Variable> input, bool training) const {
    std::cerr << "[DEBUG] Block: Entering forward" << std::endl;
    std::cerr << "[DEBUG] Block: Input device=" << (input->getData().getDevice() == Device::CUDA ? "CUDA" : "CPU") << std::endl;

    std::cerr << "[DEBUG] Block: Calling LayerNorm1" << std::endl;
    auto normed1 = norm1.forward(input);
    std::cerr << "[DEBUG] Block: LayerNorm1 done, output device=" << (normed1->getData().getDevice() == Device::CUDA ? "CUDA" : "CPU") << std::endl;

    std::cerr << "[DEBUG] Block: Calling Attention" << std::endl;
    auto attention_output = attention.forward(normed1, training);
    std::cerr << "[DEBUG] Block: Attention done, output device=" << (attention_output->getData().getDevice() == Device::CUDA ? "CUDA" : "CPU") << std::endl;

    std::cerr << "[DEBUG] Block: Computing residual connection 1" << std::endl;
    auto residual1 = input->add(attention_output);
    std::cerr << "[DEBUG] Block: Residual1 done" << std::endl;

    std::cerr << "[DEBUG] Block: Calling LayerNorm2" << std::endl;
    auto normed2 = norm2.forward(residual1);
    std::cerr << "[DEBUG] Block: LayerNorm2 done" << std::endl;

    std::cerr << "[DEBUG] Block: Calling FFN" << std::endl;
    auto ffn_output = ffn.forward(normed2, training);
    std::cerr << "[DEBUG] Block: FFN done" << std::endl;

    std::cerr << "[DEBUG] Block: Computing residual connection 2" << std::endl;
    auto result = residual1->add(ffn_output);
    std::cerr << "[DEBUG] Block: Forward complete" << std::endl;
    return result;
}