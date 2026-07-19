#pragma once
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
    public:
        // modern = true assembles the Llama-style block: RMSNorm, RoPE
        // attention, and a bias-free SwiGLU FFN. Default is the original
        // GPT-2-style block (LayerNorm, learned positions, GELU FFN).
        TransformerBlock(int d_model, int num_heads, int ffn_hidden_dim = -1,
                         float dropout_rate = 0.1f, bool modern = false);
        std::shared_ptr<Variable> forward(std::shared_ptr<Variable> input, bool training = false) const;

        const MultiHeadAttention& getAttention() const { return attention; }
        const FeedForward& getFFN() const { return ffn; }
        const LayerNorm& getNorm1() const { return norm1; }
        const LayerNorm& getNorm2() const { return norm2; }
        
        MultiHeadAttention& getAttentionRef() { return attention; }
        FeedForward& getFeedForwardRef() { return ffn; }
        LayerNorm& getNorm1Ref() { return norm1; }
        LayerNorm& getNorm2Ref() { return norm2; }
};