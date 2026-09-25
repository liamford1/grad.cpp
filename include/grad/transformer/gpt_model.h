#pragma once
#include "grad/transformer/tensor.h"
#include "grad/transformer/token_embedding.h"
#include "grad/transformer/positional_encoding.h"
#include "grad/transformer/transformer_block.h"
#include "grad/transformer/linear.h"
#include "grad/transformer/layer_norm.h"

#include <vector>
#include <memory>
#include <string>

// GPT2: LayerNorm, learned absolute positional embeddings, GELU FFN with
// biases. Modern: RMSNorm, RoPE, bias-free SwiGLU FFN - same parameter
// count at equal d_model. The arch is stored in the checkpoint (format v2;
// v1 files load as GPT2).
enum class GPTArch { GPT2 = 0, Modern = 1 };

class GPTModel {
    private:
        int vocab_size;
        int d_model;
        int num_layers;
        int num_heads;
        int max_len;
        float dropout_rate;
        GPTArch arch;

        TokenEmbedding token_embedding;
        // Constructed for both arches to keep the class layout simple, but
        // unused by Modern: forward skips it and it is excluded from
        // getAllParameters() and the checkpoint (RoPE replaces it).
        PositionalEncoding pos_encoding;
        std::vector<std::unique_ptr<TransformerBlock>> transformer_blocks;
        LayerNorm final_norm;
    public:
        GPTModel(int vocab_size, int d_model, int num_layers, int num_heads, int max_len,
                 float dropout_rate = 0.1f, GPTArch arch = GPTArch::GPT2);
        ~GPTModel() = default;

        [[nodiscard]] std::shared_ptr<Variable> forward(std::shared_ptr<Variable> token_ids, bool training = false) const;

        [[nodiscard]] std::vector<std::shared_ptr<Variable>> getAllParameters() const;

        int getVocabSize() const { return vocab_size; }
        int getDModel() const { return d_model; }
        int getNumLayers() const { return num_layers; }
        int getNumHeads() const { return num_heads; }
        int getMaxLen() const { return max_len; }
        GPTArch getArch() const { return arch; }

        const TokenEmbedding& getTokenEmbedding() const { return token_embedding; }
        const PositionalEncoding& getPosEncoding() const { return pos_encoding; }
        const TransformerBlock& getBlock(int i) const { return *transformer_blocks[i]; }
        const LayerNorm& getFinalNorm() const { return final_norm; }

        // quiet suppresses the success print - used for the periodic
        // resume-state writes, which would otherwise log a .tmp filename
        // every eval interval.
        [[nodiscard]] bool save(const std::string& filepath, bool quiet = false) const;
        [[nodiscard]] static GPTModel load(const std::string& filepath);

        GPTModel(const GPTModel&) = delete;
        GPTModel& operator=(const GPTModel&) = delete;
        
        GPTModel(GPTModel&&) = default;
        GPTModel& operator=(GPTModel&&) = default;
};