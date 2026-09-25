#pragma once

#include "grad/transformer/gpt_model.h"
#include <vector>

namespace grad {

// Incremental decoding with a KV cache.
//
// The autograd path re-runs the whole prefix through the model for every
// generated token, which is O(context^2) forward work per token. This
// session instead processes one token at a time against cached per-layer
// K/V projections, so each step costs O(context * d_model) attention plus
// the fixed per-token matmuls. It works on raw float buffers - inference
// needs neither gradients nor graph bookkeeping.
class InferenceSession {
public:
    explicit InferenceSession(const GPTModel& model);

    // Appends token_id to the context and returns the logits for the next
    // token: vocab_size floats, valid until the next step() call.
    // Throws if the context would exceed the model's max_len.
    [[nodiscard]] const float* step(int token_id);

    // In the model's int hyperparameter terms; pos_ < max_len_, so all
    // three fit.
    int position() const { return static_cast<int>(pos_); }
    int capacity() const { return static_cast<int>(max_len_); }
    int vocabSize() const { return static_cast<int>(vocab_); }

private:
    const GPTModel& model_;
    size_t d_model_;
    size_t num_layers_;
    size_t num_heads_;
    size_t head_size_;
    size_t d_ff_;
    size_t max_len_;
    size_t vocab_;
    size_t pos_ = 0;

    // Per layer: (max_len, d_model) rows of projected K and V.
    std::vector<std::vector<float>> k_cache_;
    std::vector<std::vector<float>> v_cache_;

    // Scratch buffers reused across steps.
    std::vector<float> x_;           // residual stream (d_model)
    std::vector<float> h_;           // normed input / final norm out (d_model)
    std::vector<float> q_;           // query projection (d_model)
    std::vector<float> attn_;        // concatenated head outputs (d_model)
    std::vector<float> scores_;      // attention scores (max_len)
    std::vector<float> ffn_hidden_;  // FFN activation (d_ff)
    std::vector<float> ffn_gate_;    // SwiGLU gate activation (d_ff, modern)
    std::vector<float> rope_cos_;    // per-step RoPE angles (head_size/2, modern)
    std::vector<float> rope_sin_;
    std::vector<float> logits_;  // output logits (vocab)
};

}  // namespace grad
