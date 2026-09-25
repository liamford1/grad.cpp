#pragma once

#include "grad/transformer/gpt_model.h"
#include "grad/transformer/optimizer.h"
#include "grad/data/dataloader.h"
#include "grad/utils/metrics.h"
#include <string>
#include <memory>
#include <optional>

namespace grad::training {

struct TrainingConfig {
    int vocab_size;
    int d_model;
    int num_layers;
    int num_heads;
    int max_len;
    int seq_length;
    int batch_size;
    float learning_rate;
    float dropout;
    int warmup_steps;
    int num_steps;
    int checkpoint_interval;
    std::string checkpoint_prefix;
    // Micro-batches accumulated per optimizer step. Effective batch size is
    // batch_size * grad_accum, but peak memory stays that of one
    // micro-batch: each micro-batch's graph is built, backpropagated, and
    // released before the next one starts.
    int grad_accum = 1;
    int eval_interval = 250;    // steps between validation runs (0 = never)
    float weight_decay = 0.1f;  // AdamW decay on weight matrices
    // Cap on batches per validation pass (0 = whole val set). At large
    // corpus scale the 5% holdout is tens of millions of tokens; a fixed
    // set of windows spread evenly across it (SpreadSubset) gives a stable,
    // representative perplexity estimate in bounded time.
    int max_eval_batches = 0;
};

class Trainer {
public:
    Trainer(const TrainingConfig& config,
            GPTModel& model,
            DataLoader& loader,
            DataLoader* val_loader = nullptr);

    // Returns false when the run was interrupted (SIGINT/SIGTERM): resume
    // state has been saved and the caller should skip end-of-run work.
    [[nodiscard]] bool train();
    // Writes the model to path via a temp file and rename, so an existing
    // checkpoint is only ever replaced by a complete one. False on failure
    // (after a warning); the previous file, if any, is left untouched.
    [[nodiscard]] bool save_checkpoint(const std::string& path);

    // Mean loss over the validation set, or with max_eval_batches > 0 over
    // that many batches of windows spread evenly across it, the same ones
    // on every call (forward-only, no dropout). Perplexity is exp of this.
    // Returns -1 if there is no val loader.
    [[nodiscard]] float evaluate();

    // Restores the state written by save_resume_state: optimizer moments,
    // step position, and best val loss. The model weights themselves come
    // from the companion <prefix>_resume_model.bin, which the caller loads
    // before constructing the Trainer. Returns false if the state file is
    // missing or does not match the current model.
    [[nodiscard]] bool load_resume_state();

private:
    TrainingConfig config_;
    GPTModel& model_;
    DataLoader& loader_;
    DataLoader* val_loader_;
    // Capped-eval view of val_loader_'s dataset; null when evaluating the
    // whole split.
    std::unique_ptr<DataLoader> eval_loader_;
    std::unique_ptr<AdamOptimizer> optimizer_;
    std::unique_ptr<utils::TrainingMetrics> metrics_;
    std::unique_ptr<utils::MetricsLog> mlog_;
    int start_step_ = 0;
    float best_val_loss_restored_ = -1.0f;  // <0 = no restored value

    void training_step(int step);
    std::string resume_model_path() const;
    std::string resume_state_path() const;
    // False (after a warning) if either file of the pair failed to write.
    [[nodiscard]] bool save_resume_state(int next_step, float best_val_loss);
};

// Mean next-token cross-entropy (nats) over up to max_batches batches of
// loader, from its current position (0 = until the loader is exhausted).
// Forward-only with dropout off; each batch's graph is released before the
// next is built. Batches are weighted by row count, so a short final batch
// does not skew the mean. Returns -1 if the loader yields nothing.
[[nodiscard]] double mean_loss(GPTModel& model, DataLoader& loader, int max_batches = 0);

// Reads just the step position from a resume state file, so callers can
// derive run parameters (e.g. the data loader seed) before the Trainer
// exists. Empty if the file is missing or is not a resume state file.
[[nodiscard]] std::optional<int> peek_resume_step(const std::string& state_path);

}  // namespace grad::training
