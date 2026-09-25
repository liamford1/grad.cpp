#pragma once

// Named model/training configurations for `grad train` and `grad train-fast`.
// benchmarks/pytorch_baseline.py mirrors the model fields of this table;
// CI runs benchmarks/check_presets.py against `grad presets --json` so the
// two cannot drift apart silently.

#include "grad/transformer/gpt_model.h"

#include <span>
#include <string_view>

namespace cli {

struct Preset {
    std::string_view name;
    std::string_view purpose;  // one line for `grad presets`

    // Model
    int vocab_size;
    int d_model;
    int num_layers;
    int num_heads;
    int max_len;
    bool modern;  // GPTArch::Modern (RMSNorm, RoPE, SwiGLU) instead of GPT-2

    // Batching: effective batch is batch_size * grad_accum sequences of
    // seq_length tokens.
    int seq_length;
    int batch_size;
    int grad_accum;

    // Optimization
    float learning_rate;
    float dropout;
    int warmup_steps;
    int num_steps;

    // Checkpointing and validation
    int checkpoint_interval;
    int eval_interval;
    int max_eval_batches;  // 0 = evaluate the whole val set

    [[nodiscard]] GPTArch arch() const noexcept { return modern ? GPTArch::Modern : GPTArch::GPT2; }
    // The "fast" presets are smoke tests; their checkpoints get a "_fast"
    // prefix so they never clobber a real run on the same corpus.
    [[nodiscard]] bool is_smoke_test() const noexcept { return name.starts_with("fast"); }
};

// Every preset, in display order.
[[nodiscard]] std::span<const Preset> presets() noexcept;

// The preset called name, or nullptr.
[[nodiscard]] const Preset* find_preset(std::string_view name) noexcept;

}  // namespace cli
