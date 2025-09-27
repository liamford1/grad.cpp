#include "transformer/tensor.h"
#include "transformer/token_embedding.h"
#include "transformer/positional_encoding.h"
#include "transformer/transformer_block.h"
#include "transformer/linear.h"
#include "transformer/layer_norm.h"
#include "transformer/gpt_model.h"
#include <iostream>
#include <memory>
#include <vector>

GPTModel::GPTModel(int vocab_size, int d_model, int num_layers, int num_heads, int max_len, float dropout_rate) : 
    vocab_size(vocab_size),
    d_model(d_model),
    num_layers(num_layers),
    num_heads(num_heads),
    max_len(max_len),
    dropout_rate(dropout_rate),
    token_embedding(vocab_size, d_model),
    pos_encoding(max_len, d_model),
    final_norm(d_model),
    output_projection(d_model, vocab_size)
{
    for (int i = 0; i < num_layers; i++) {
        transformer_blocks.push_back(std::make_unique<TransformerBlock>(d_model, num_heads, -1, dropout_rate));
    }
}

Tensor GPTModel::forward(const Tensor& token_ids, bool training) const {
    Tensor embed_tokens = token_embedding.forward(token_ids);
    Tensor encode_positions = pos_encoding.forward(embed_tokens);

    Tensor transformer_output = encode_positions;

    for (int i = 0; i < num_layers; i++) {
        transformer_output = transformer_blocks[i]->forward(transformer_output, training);
    }

    Tensor normalized_output = final_norm.forward(transformer_output);
    
    Tensor logits = output_projection.forward(normalized_output);
    return logits;
}