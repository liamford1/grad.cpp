#include "presets.h"

#include <array>

namespace grad::cli {

namespace {

// "small" is the original 22M-param Shakespeare config; "fast" is the CI
// smoke test.
//
// small's step count is set by measurement: on the 254K-token Shakespeare
// corpus, val perplexity bottoms around step 6000 and rises after (a 22M
// model memorizes a corpus that small). 8000 steps lets the cosine
// schedule finish near the minimum instead of training 8x past it.
//
// "medium" (~70M params) is memory-bound, not compute-bound: one step at
// batch 16 / seq 256 once peaked past 10GB of phys footprint and took a
// 16GB machine down (measured 2026-07-18, BENCHMARKS.md #8). Micro-batch
// 8 with grad_accum 4 gives the optimizer an effective batch of 32 at a
// ~4GB peak (lazy grads + backward retirement, BENCHMARKS.md #9). The
// micro-batch is also measured, not assumed: 16 x accum 2 - same
// effective batch, FFN matmuls above the Metal crossover - ran 32%
// slower per token and 3GB fatter, with or without the GPU. At 40000
// steps the run consumes 40000*32*256 = 327M tokens, about one epoch of
// the TinyStories train split.
//
// "modern" is "medium" with the Llama-style block (RMSNorm, RoPE, SwiGLU;
// see gpt_model.h) at the same parameter count and training budget, so the
// two runs A/B the architecture and nothing else. Its checkpoints get a
// "_modern" prefix so both lineages can coexist on one corpus.
//
// Dropout is per preset because it answers a per-corpus question. small
// repeats its 254K-token corpus many times over and memorizes it, so 0.1
// earns its keep. medium/modern see 327M tokens of a 363M-token corpus,
// under one epoch: there is no repeated exposure to regularize against,
// and the mask generation was the largest non-BLAS cost in the profile
// (13% of compute). Single-epoch pretraining runs use 0 for this reason.

constexpr Preset kFast{
    .name = "fast",
    .purpose = "50-step smoke test (train-fast)",
    .vocab_size = 500,
    .d_model = 128,
    .num_layers = 2,
    .num_heads = 4,
    .max_len = 1024,
    .modern = false,
    .seq_length = 64,
    .batch_size = 4,
    .grad_accum = 1,
    .learning_rate = 3e-4f,
    .dropout = 0.1f,
    .warmup_steps = 10,
    .num_steps = 50,
    .checkpoint_interval = 2500,
    .eval_interval = 25,
    .max_eval_batches = 0,
};

constexpr Preset kSmall{
    .name = "small",
    .purpose = "Tiny Shakespeare (the default)",
    .vocab_size = 5000,
    .d_model = 512,
    .num_layers = 6,
    .num_heads = 8,
    .max_len = 1024,
    .modern = false,
    .seq_length = 96,
    .batch_size = 8,
    .grad_accum = 1,
    .learning_rate = 3e-4f,
    .dropout = 0.1f,
    .warmup_steps = 500,
    .num_steps = 8000,
    .checkpoint_interval = 2500,
    .eval_interval = 250,
    .max_eval_batches = 0,
};

constexpr Preset kMedium{
    .name = "medium",
    .purpose = "TinyStories-scale corpora",
    .vocab_size = 16000,
    .d_model = 768,
    .num_layers = 8,
    .num_heads = 12,
    .max_len = 1024,
    .modern = false,
    .seq_length = 256,
    .batch_size = 8,
    .grad_accum = 4,
    .learning_rate = 3e-4f,
    .dropout = 0.0f,
    .warmup_steps = 1000,
    .num_steps = 40000,
    .checkpoint_interval = 4000,
    .eval_interval = 500,
    .max_eval_batches = 32,
};

// The modern-architecture twins differ from their GPT-2 originals in the
// block alone.
constexpr Preset with_modern_block(Preset preset, std::string_view name,
                                   std::string_view purpose) {
    preset.name = name;
    preset.purpose = purpose;
    preset.modern = true;
    return preset;
}

constexpr std::array kPresets{
    kFast,
    with_modern_block(kFast, "fast-modern", "smoke test of the modern block"),
    kSmall,
    kMedium,
    with_modern_block(kMedium, "modern", "architecture A/B against medium"),
};

}  // namespace

std::span<const Preset> presets() noexcept {
    return kPresets;
}

const Preset* find_preset(std::string_view name) noexcept {
    for (const Preset& preset : kPresets) {
        if (preset.name == name) return &preset;
    }
    return nullptr;
}

}  // namespace grad::cli
