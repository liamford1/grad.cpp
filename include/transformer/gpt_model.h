#pragma once
#include "tensor.h"
#include "token_embedding.h"
#include "positional_encoding.h"
#include "transformer_block.h"
#include "linear.h"
#include "layer_norm.h"

#include <vector>
#include <memory>
#include <fstream>
#include <iostream>
#include <string>

class GPTModel {
    private:
        int vocab_size;
        int d_model;
        int num_layers;
        int num_heads;
        int max_len;
        float dropout_rate;

        TokenEmbedding token_embedding;
        PositionalEncoding pos_encoding;
        std::vector<std::unique_ptr<TransformerBlock>> transformer_blocks;
        LayerNorm final_norm;
    public:
        GPTModel(int vocab_size, int d_model, int num_layers, int num_heads, int max_len, float dropout_rate = 0.1f);
        ~GPTModel() = default;

        std::shared_ptr<Variable> forward(std::shared_ptr<Variable> token_ids, bool training = false) const;

        std::vector<std::shared_ptr<Variable>> getAllParameters() const;

        int getVocabSize() const { return vocab_size; }
        int getDModel() const { return d_model; }
        int getNumLayers() const { return num_layers; }

        bool save(const std::string& filepath) const;
        static GPTModel load(const std::string& filepath);

        GPTModel(const GPTModel&) = delete;
        GPTModel& operator=(const GPTModel&) = delete;
        
        GPTModel(GPTModel&&) = default;
        GPTModel& operator=(GPTModel&&) = default;
};