#pragma once

#include "transformer/gpt_model.h"
#include "transformer/optimizer.h"
#include "data/dataloader.h"
#include "tokenizer/bpe_tokenizer.h"
#include "utils/metrics.h"
#include <string>
#include <memory>

namespace training {

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
    // sample of the (unshuffled) val loader gives a stable perplexity
    // estimate in bounded time.
    int max_eval_batches = 0;
};

class Trainer {
public:
    Trainer(const TrainingConfig& config,
            GPTModel& model,
            DataLoader& loader,
            BPETokenizer& tokenizer,
            DataLoader* val_loader = nullptr);

    // Returns false when the run was interrupted (SIGINT/SIGTERM): resume
    // state has been saved and the caller should skip end-of-run work.
    bool train();
    void save_checkpoint(const std::string& path);

    // Mean loss over the whole validation set (forward-only, no dropout).
    // Perplexity is exp of this. Returns -1 if there is no val loader.
    float evaluate();

    // Restores the state written by save_resume_state: optimizer moments,
    // step position, and best val loss. The model weights themselves come
    // from the companion <prefix>_resume_model.bin, which the caller loads
    // before constructing the Trainer. Returns false if the state file is
    // missing or does not match the current model.
    bool load_resume_state();

private:
    TrainingConfig config_;
    GPTModel& model_;
    DataLoader& loader_;
    BPETokenizer& tokenizer_;
    DataLoader* val_loader_;
    std::unique_ptr<AdamOptimizer> optimizer_;
    std::unique_ptr<utils::TrainingMetrics> metrics_;
    std::unique_ptr<utils::MetricsLog> mlog_;
    int start_step_ = 0;
    float best_val_loss_restored_ = -1.0f;  // <0 = no restored value

    void training_step(int step);
    std::string resume_model_path() const;
    std::string resume_state_path() const;
    void save_resume_state(int next_step, float best_val_loss);
    void log_performance_breakdown();
};

// Reads just the step position from a resume state file, so callers can
// derive run parameters (e.g. the data loader seed) before the Trainer
// exists. Returns -1 if the file is missing or not a resume state file.
int peek_resume_step(const std::string& state_path);

} // namespace training
