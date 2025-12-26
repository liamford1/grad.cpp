#include "training/trainer.h"
#include "utils/training_utils.h"
#include "transformer/variable.h"
#include <iostream>
#include <iomanip>
#include <chrono>

namespace training {

Trainer::Trainer(const TrainingConfig& config,
                 GPTModel& model,
                 DataLoader& loader,
                 BPETokenizer& tokenizer)
    : config_(config), model_(model), loader_(loader), tokenizer_(tokenizer) {

    auto params = model_.getAllParameters();
    optimizer_ = std::make_unique<AdamOptimizer>(params, config_.learning_rate,
                                                  0.9f, 0.999f, 1e-8f, 0.0f);
    optimizer_->set_warmup_steps(config_.warmup_steps);

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
    std::cout << "  Learning rate: " << config_.learning_rate << std::endl;
    std::cout << "  Training steps: " << config_.num_steps << "\n" << std::endl;

    std::cout << std::setw(10) << "Step"
              << std::setw(15) << "Loss"
              << std::setw(15) << "Grad Norm" << std::endl;
    std::cout << std::string(40, '-') << std::endl;

    metrics_->start_training();

    for (int step = 0; step < config_.num_steps; step++) {
        training_step(step);

        if (step > 0 && step % config_.checkpoint_interval == 0) {
            std::string checkpoint_path = config_.checkpoint_prefix + "_step_" + std::to_string(step) + ".bin";
            save_checkpoint(checkpoint_path);
            std::cout << "  [checkpoint: " << checkpoint_path << "]" << std::endl;
        }
    }

    metrics_->print_summary();
    save_checkpoint(config_.checkpoint_prefix + "_final.bin");
}

void Trainer::training_step(int step) {
    if (!loader_.has_next()) loader_.reset();
    auto batch = loader_.next_batch();

    int batch_size = batch.input.getBatchSize();
    int seq_len = batch.input.getRows();

    Tensor input_2d(batch_size * seq_len, 1);
    Tensor target_2d(batch_size * seq_len, 1);
    utils::reshape_batch_to_2d(batch.input, batch.target, input_2d, target_2d);

    auto in = Variable::create(input_2d, false);
    auto tgt = Variable::create(target_2d, false);

    std::cerr << "[DEBUG] Step " << step << ": Starting Forward" << std::endl;
    auto logits = model_.forward(in, true);
    auto loss = logits->log_softmax()->nll_loss(tgt);
    float loss_val = loss->getData().getValue(0, 0);
    std::cerr << "[DEBUG] Step " << step << ": Forward Done, Loss = " << loss_val << std::endl;

    std::cerr << "[DEBUG] Step " << step << ": Zeroing Gradients" << std::endl;
    optimizer_->zero_grad();

    std::cerr << "[DEBUG] Step " << step << ": Starting Backward" << std::endl;
    loss->backward();
    std::cerr << "[DEBUG] Step " << step << ": Backward Done" << std::endl;

    loss->release_graph();

    std::cerr << "[DEBUG] Step " << step << ": Starting Optimizer" << std::endl;
    optimizer_->clip_grad_norm(5.0f);
    optimizer_->step();
    std::cerr << "[DEBUG] Step " << step << ": Optimizer Done" << std::endl;

    if (step % 10 == 0) {
        float loss_val = loss->getData().getValue(0, 0);
        metrics_->print_progress(step, loss_val);

        if (step % 100 == 0) {
            auto params = model_.getAllParameters();
            float grad_norm = utils::compute_grad_norm(params);
            metrics_->record_step(step, loss_val, grad_norm);
        }
    }
}

void Trainer::save_checkpoint(const std::string& path) {
    model_.save(path);
}

} // namespace training
