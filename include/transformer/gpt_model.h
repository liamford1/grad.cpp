#ifndef GPT_MODEL_H
#define GPT_MODEL_H

#include "tensor.h"
#include "token_embedding.h"
#include "positional_encoding.h"
#include "transformer_block.h"
#include "linear.h"
#include "layer_norm.h"

#include <vector>
#include <memory>

class GPTModel {
    private:
        int vocab_size;
        int d_model;
        int num_layers;
        int num_heads;
        int max_len;

        TokenEmbedding token_embedding;
        PositionalEncoding pos_encoding;
        std::vector<std::unique_ptr<TransformerBlock>> transformer_blocks;
        LayerNorm final_norm;
        Linear output_projection;
    public:
        GPTModel(int vocab_size, int d_model, int num_layers, int num_heads, int max_len);
        ~GPTModel() = default;
        Tensor forward(const Tensor& token_ids, bool training = false) const;
};

#endif