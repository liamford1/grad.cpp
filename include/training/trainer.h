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
    int eval_interval = 250;    // steps between validation runs (0 = never)
    float weight_decay = 0.1f;  // AdamW decay on weight matrices

};

class Trainer {
public:
    Trainer(const TrainingConfig& config,
            GPTModel& model,
            DataLoader& loader,
            BPETokenizer& tokenizer,
            DataLoader* val_loader = nullptr);

    void train();
    void save_checkpoint(const std::string& path);

    // Mean loss over the whole validation set (forward-only, no dropout).
    // Perplexity is exp of this. Returns -1 if there is no val loader.
    float evaluate();

private:
    TrainingConfig config_;
    GPTModel& model_;
    DataLoader& loader_;
    BPETokenizer& tokenizer_;
    DataLoader* val_loader_;
    std::unique_ptr<AdamOptimizer> optimizer_;
    std::unique_ptr<utils::TrainingMetrics> metrics_;

    void training_step(int step);
    void log_performance_breakdown();
};

} // namespace training
