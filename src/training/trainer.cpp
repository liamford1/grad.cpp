#include "training/trainer.h"
#include "utils/training_utils.h"
#include "transformer/variable.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include <chrono>

namespace training {

Trainer::Trainer(const TrainingConfig& config,
                 GPTModel& model,
                 DataLoader& loader,
                 BPETokenizer& tokenizer,
                 DataLoader* val_loader)
    : config_(config), model_(model), loader_(loader), tokenizer_(tokenizer),
      val_loader_(val_loader) {

    auto params = model_.getAllParameters();
    optimizer_ = std::make_unique<AdamOptimizer>(params, config_.learning_rate,
                                                  0.9f, 0.999f, 1e-8f,
                                                  config_.weight_decay);
    // Linear warmup, then cosine decay to 10% of the base learning rate.
    optimizer_->set_schedule(config_.warmup_steps, config_.num_steps,
                             config_.learning_rate * 0.1f);

    metrics_ = std::make_unique<utils::TrainingMetrics>(config_.num_steps);
}

void Trainer::train() {
    utils::print_header("Training Started");

    std::cout << "Config:" << std::endl;
    std::cout << "  Vocab size: " << config_.vocab_size << std::endl;
    std::cout << "  Model: d_model=" << config_.d_model
              << " layers=" << config_.num_layers
              << " heads=" << config_.num_heads << std::endl;
    std::cout << "  Sequence length: " << config_.seq_length << std::endl;
    std::cout << "  Batch size: " << config_.batch_size << std::endl;
    // Explicit format: inherited stream state (progress printers set
    // fixed(1)) would render 3e-4 as "0.0".
    std::cout << "  Learning rate: " << std::defaultfloat << std::setprecision(6)
              << config_.learning_rate << std::endl;
    std::cout << "  Training steps: " << config_.num_steps << "\n" << std::endl;

    std::cout << std::setw(10) << "Step"
              << std::setw(15) << "Loss"
              << std::setw(15) << "Grad Norm" << std::endl;
    std::cout << std::string(40, '-') << std::endl;

    metrics_->start_training();

    // Best-val checkpointing: training loss keeps falling long after the
    // model starts memorizing (observed on Shakespeare: val perplexity
    // bottomed at step ~6000 of 50000, then quintupled). The checkpoint
    // worth keeping is the val-loss minimum, saved as it happens.
    float best_val_loss = std::numeric_limits<float>::max();

    for (int step = 0; step < config_.num_steps; step++) {
        training_step(step);

        if (val_loader_ && config_.eval_interval > 0
            && step > 0 && step % config_.eval_interval == 0) {
            float val_loss = evaluate();
            bool improved = val_loss < best_val_loss;
            std::cout << "  [step " << step
                      << " | val loss " << std::fixed << std::setprecision(4) << val_loss
                      << " | perplexity " << std::setprecision(1) << std::exp(val_loss)
                      << (improved ? " | best]" : "]") << std::defaultfloat << std::endl;
            if (improved) {
                best_val_loss = val_loss;
                save_checkpoint(config_.checkpoint_prefix + "_best.bin");
            }
        }

        if (step > 0 && step % config_.checkpoint_interval == 0) {
            std::string checkpoint_path = config_.checkpoint_prefix + "_step_" + std::to_string(step) + ".bin";
            save_checkpoint(checkpoint_path);
            std::cout << "  [checkpoint: " << checkpoint_path << "]" << std::endl;
        }
    }

    metrics_->print_summary();
    if (val_loader_) {
        float val_loss = evaluate();
        std::cout << "Final val loss: " << val_loss
                  << " | perplexity: " << std::exp(val_loss) << std::endl;
        if (best_val_loss < std::numeric_limits<float>::max()) {
            std::cout << "Best val loss: " << best_val_loss
                      << " | perplexity: " << std::exp(best_val_loss)
                      << " (saved as " << config_.checkpoint_prefix << "_best.bin)" << std::endl;
        }
    }
    save_checkpoint(config_.checkpoint_prefix + "_final.bin");
}

float Trainer::evaluate() {
    if (!val_loader_) return -1.0f;

    val_loader_->reset();
    double total_loss = 0.0;
    int batches = 0;

    while (val_loader_->has_next()) {
        if (config_.max_eval_batches > 0 && batches >= config_.max_eval_batches) break;
        auto batch = val_loader_->next_batch();
        auto in = Variable::create(batch.input, false);
        auto tgt = Variable::create(batch.target, false);

        // Forward-only: training=false disables dropout. The graph is still
        // built (parameters require grad), so release it.
        auto logits = model_.forward(in, false);
        auto loss = logits->log_softmax()->nll_loss(tgt);
        total_loss += loss->getData().getValue(0, 0);
        loss->release_graph();
        batches++;
    }
    return batches > 0 ? static_cast<float>(total_loss / batches) : -1.0f;
}

void Trainer::training_step(int step) {
    if (!loader_.has_next()) loader_.reset();
    auto batch = loader_.next_batch();

    // Feed the batch as (batch, seq, 1) so attention runs per sequence.
    // Flattening to 2D concatenates the batch into one long sequence:
    // tokens attend across sequence boundaries and the attention matrix
    // grows from batch*seq^2 to (batch*seq)^2.
    auto in = Variable::create(batch.input, false);
    auto tgt = Variable::create(batch.target, false);

    auto logits = model_.forward(in, true);
    auto loss = logits->log_softmax()->nll_loss(tgt);

    optimizer_->zero_grad();
    loss->backward();
    loss->release_graph();

    float loss_val = loss->getData().getValue(0, 0);
    float grad_norm = 0.0f;

    if (step % 100 == 0) {
        auto params = model_.getAllParameters();
        grad_norm = utils::compute_grad_norm(params);
        metrics_->record_step(step, loss_val, grad_norm);
    }

    optimizer_->clip_grad_norm(5.0f);
    optimizer_->step();

    if (step % 10 == 0) {
        metrics_->print_progress(step, loss_val);
    }
}

void Trainer::save_checkpoint(const std::string& path) {
    model_.save(path);
}

} // namespace training
