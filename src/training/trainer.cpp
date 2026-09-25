#include "training/trainer.h"
#include "utils/training_utils.h"
#include "transformer/variable.h"
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <signal.h>

namespace training {

namespace {

constexpr uint32_t kResumeMagic = 0x54524E31;  // "TRN1"
constexpr uint32_t kResumeVersion = 1;

// SIGINT/SIGTERM request a stop; the training loop honors it at the next
// step boundary so the resume state is always written from a consistent
// point. SA_RESETHAND restores the default action once a signal has been
// delivered, so a second Ctrl-C force-kills as usual. The handler only sets
// a flag; C++17 lets a handler call std::signal only for the signal being
// handled, so it cannot reset the other one itself.
volatile std::sig_atomic_t g_stop_requested = 0;

void request_stop(int) {
    g_stop_requested = 1;
}

// Installs request_stop for SIGINT and SIGTERM for its lifetime and puts the
// previous handlers back on destruction, including when train() throws.
class StopSignalGuard {
public:
    StopSignalGuard() {
        g_stop_requested = 0;
        struct sigaction action {};
        action.sa_handler = request_stop;
        sigemptyset(&action.sa_mask);
        action.sa_flags = SA_RESETHAND;
        sigaction(SIGINT, &action, &prev_int_);
        sigaction(SIGTERM, &action, &prev_term_);
    }
    ~StopSignalGuard() {
        sigaction(SIGINT, &prev_int_, nullptr);
        sigaction(SIGTERM, &prev_term_, nullptr);
    }
    StopSignalGuard(const StopSignalGuard&) = delete;
    StopSignalGuard& operator=(const StopSignalGuard&) = delete;

private:
    struct sigaction prev_int_ {};
    struct sigaction prev_term_ {};
};

// rename() replaces the destination atomically within a filesystem, so a
// reader (or a crash) sees either the old file or the complete new one.
bool replace_file(const std::string& tmp, const std::string& path) {
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::cerr << "Warning: failed to rename " << tmp << " -> " << path
                  << ": " << std::strerror(errno) << std::endl;
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

}  // namespace

std::optional<int> peek_resume_step(const std::string& state_path) {
    std::ifstream in(state_path, std::ios::binary);
    if (!in.is_open()) return std::nullopt;
    uint32_t magic, version;
    int32_t next_step;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&next_step), sizeof(next_step));
    if (!in.good() || magic != kResumeMagic || version != kResumeVersion) return std::nullopt;
    return next_step;
}

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

std::string Trainer::resume_model_path() const {
    return config_.checkpoint_prefix + "_resume_model.bin";
}

std::string Trainer::resume_state_path() const {
    return config_.checkpoint_prefix + "_resume_state.bin";
}

// Writes the model + optimizer/trainer state pair. Both files are fully
// written to .tmp siblings before either is renamed into place, so a failed
// or interrupted write never replaces a good file with a partial one. The
// state file is renamed last because its step decides where `resume`
// continues: the one unguarded window (a crash between the two renames)
// pairs new weights with the previous step and optimizer state, which
// replays those steps rather than skipping any.
bool Trainer::save_resume_state(int next_step, float best_val_loss) {
    const std::string model_tmp = resume_model_path() + ".tmp";
    const std::string state_tmp = resume_state_path() + ".tmp";

    if (!model_.save(model_tmp, /*quiet=*/true)) {
        std::cerr << "Warning: failed to write " << model_tmp << std::endl;
        std::remove(model_tmp.c_str());
        return false;
    }

    bool state_ok = false;
    {
        std::ofstream out(state_tmp, std::ios::binary);
        int32_t step32 = next_step;
        out.write(reinterpret_cast<const char*>(&kResumeMagic), sizeof(kResumeMagic));
        out.write(reinterpret_cast<const char*>(&kResumeVersion), sizeof(kResumeVersion));
        out.write(reinterpret_cast<const char*>(&step32), sizeof(step32));
        out.write(reinterpret_cast<const char*>(&best_val_loss), sizeof(best_val_loss));
        state_ok = optimizer_->save_state(out) && out.good();
        out.close();
        state_ok = state_ok && !out.fail();
    }
    if (!state_ok) {
        std::cerr << "Warning: failed to write " << state_tmp << std::endl;
        std::remove(model_tmp.c_str());
        std::remove(state_tmp.c_str());
        return false;
    }

    if (!replace_file(model_tmp, resume_model_path())) {
        std::remove(state_tmp.c_str());
        return false;
    }
    if (!replace_file(state_tmp, resume_state_path())) {
        std::cerr << "Warning: " << resume_model_path() << " was updated but "
                  << resume_state_path() << " was not; a resume would replay steps "
                  << "from the older state's position" << std::endl;
        return false;
    }
    return true;
}

bool Trainer::load_resume_state() {
    std::ifstream in(resume_state_path(), std::ios::binary);
    if (!in.is_open()) return false;

    uint32_t magic, version;
    int32_t next_step;
    float best_val_loss;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&next_step), sizeof(next_step));
    in.read(reinterpret_cast<char*>(&best_val_loss), sizeof(best_val_loss));
    if (!in.good() || magic != kResumeMagic || version != kResumeVersion) return false;

    if (!optimizer_->load_state(in)) return false;

    start_step_ = next_step;
    best_val_loss_restored_ = best_val_loss;
    return true;
}

bool Trainer::train() {
    utils::print_header(start_step_ > 0 ? "Training Resumed" : "Training Started");

    std::cout << "Config:" << std::endl;
    std::cout << "  Vocab size: " << config_.vocab_size << std::endl;
    std::cout << "  Model: d_model=" << config_.d_model
              << " layers=" << config_.num_layers
              << " heads=" << config_.num_heads << std::endl;
    std::cout << "  Sequence length: " << config_.seq_length << std::endl;
    if (config_.grad_accum > 1) {
        std::cout << "  Batch size: " << config_.batch_size << " x "
                  << config_.grad_accum << " accum (effective "
                  << config_.batch_size * config_.grad_accum << ")" << std::endl;
    } else {
        std::cout << "  Batch size: " << config_.batch_size << std::endl;
    }
    // Explicit format: inherited stream state (progress printers set
    // fixed(1)) would render 3e-4 as "0.0".
    std::cout << "  Learning rate: " << std::defaultfloat << std::setprecision(6)
              << config_.learning_rate << std::endl;
    std::cout << "  Training steps: " << config_.num_steps;
    if (start_step_ > 0) std::cout << " (resuming at " << start_step_ << ")";
    std::cout << "\n" << std::endl;

    std::cout << std::setw(10) << "Step"
              << std::setw(15) << "Loss"
              << std::setw(15) << "Grad Norm" << std::endl;
    std::cout << std::string(40, '-') << std::endl;

    metrics_->start_training(start_step_);

    // Per-step CSV for `grad watch`; resumes append so the
    // dashboard sees the run's whole history.
    long param_count = 0;
    for (const auto& p : model_.getAllParameters()) param_count += p->getData().numel();
    char desc[128];
    std::snprintf(desc, sizeof(desc), "%s d%d L%d H%d seq%d b%dx%d vocab%d",
                  model_.getArch() == GPTArch::Modern ? "modern" : "gpt2",
                  config_.d_model, config_.num_layers, config_.num_heads,
                  config_.seq_length, config_.batch_size, config_.grad_accum,
                  config_.vocab_size);
    mlog_ = std::make_unique<utils::MetricsLog>(
        config_.checkpoint_prefix + "_metrics.csv", start_step_ > 0,
        config_.num_steps,
        static_cast<long>(config_.batch_size) * config_.grad_accum * config_.seq_length,
        param_count, desc);

    // Best-val checkpointing: training loss keeps falling long after the
    // model starts memorizing (observed on Shakespeare: val perplexity
    // bottomed at step ~6000 of 50000, then quintupled). The checkpoint
    // worth keeping is the val-loss minimum, saved as it happens.
    float best_val_loss = best_val_loss_restored_ >= 0.0f
        ? best_val_loss_restored_ : std::numeric_limits<float>::max();

    bool interrupted = false;
    // The handlers cover the loop only; a Ctrl-C during the end-of-run
    // evaluation and saves takes the default action.
    std::optional<StopSignalGuard> stop_signals(std::in_place);
    for (int step = start_step_; step < config_.num_steps; step++) {
        training_step(step);

        if (val_loader_ && config_.eval_interval > 0
            && step > 0 && step % config_.eval_interval == 0) {
            float val_loss = evaluate();
            bool improved = val_loss < best_val_loss;
            std::cout << "  [step " << step
                      << " | val loss " << std::fixed << std::setprecision(4) << val_loss
                      << " | perplexity " << std::setprecision(1) << std::exp(val_loss)
                      << (improved ? " | best]" : "]") << std::defaultfloat << std::endl;
            // best_val_loss tracks what _best.bin holds, so a failed write
            // leaves the bar where it was and the next improvement retries.
            if (improved && save_checkpoint(config_.checkpoint_prefix + "_best.bin")) {
                best_val_loss = val_loss;
            }
            mlog_->log_eval(step, val_loss);
            if (!save_resume_state(step + 1, best_val_loss)) {
                std::cerr << "Warning: resume state not updated at step " << step
                          << "; a restart resumes from the previous save" << std::endl;
            }
        }

        if (step > 0 && step % config_.checkpoint_interval == 0) {
            std::string checkpoint_path = config_.checkpoint_prefix + "_step_" + std::to_string(step) + ".bin";
            if (save_checkpoint(checkpoint_path)) {
                std::cout << "  [checkpoint: " << checkpoint_path << "]" << std::endl;
            }
        }

        if (g_stop_requested) {
            std::cout << "\n\nInterrupted after step " << step << std::endl;
            if (!save_resume_state(step + 1, best_val_loss)) {
                throw std::runtime_error("interrupted after step " + std::to_string(step)
                                         + " but could not save resume state; a resume "
                                         "restarts from the previous save");
            }
            std::cout << "Resume state saved: " << resume_model_path()
                      << " + " << resume_state_path() << std::endl;
            interrupted = true;
            break;
        }
    }
    stop_signals.reset();

    if (interrupted) return false;

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
    // The final checkpoint is the run's deliverable: failing to write it is
    // an error, not a warning, and the resume state is left at the last
    // eval so a restart retrains the tail and tries again.
    if (!save_checkpoint(config_.checkpoint_prefix + "_final.bin")) {
        throw std::runtime_error("could not write " + config_.checkpoint_prefix + "_final.bin");
    }
    if (!save_resume_state(config_.num_steps, best_val_loss)) {
        std::cerr << "Warning: final resume state not saved; `resume` would retrain "
                  << "from the last eval" << std::endl;
    }
    return true;
}

double mean_loss(GPTModel& model, DataLoader& loader, int max_batches) {
    double weighted_loss = 0.0;
    size_t rows = 0;
    for (int batches = 0; loader.has_next(); batches++) {
        if (max_batches > 0 && batches >= max_batches) break;
        auto batch = loader.next_batch();
        auto in = Variable::create(batch.input, false);
        auto tgt = Variable::create(batch.target, false);

        // The graph is still built (parameters require grad), so release it.
        auto logits = model.forward(in, false);
        auto loss = logits->log_softmax()->nll_loss(tgt);
        const size_t batch_rows = batch.input.getBatchSize();
        weighted_loss += static_cast<double>(loss->getData().getValue(0, 0)) * static_cast<double>(batch_rows);
        rows += batch_rows;
        loss->release_graph();
    }
    return rows > 0 ? weighted_loss / static_cast<double>(rows) : -1.0;
}

float Trainer::evaluate() {
    if (!val_loader_) return -1.0f;
    val_loader_->reset();
    return static_cast<float>(mean_loss(model_, *val_loader_, config_.max_eval_batches));
}

void Trainer::training_step(int step) {
    auto step_start = std::chrono::steady_clock::now();
    optimizer_->zero_grad();

    // Gradient accumulation: run grad_accum micro-batches, each building
    // and releasing its own graph, so peak memory stays that of a single
    // batch while gradients sum in the parameter tensors. The sum is
    // scaled to a mean afterwards (see scale_grads for why not through
    // the loss node).
    float loss_sum = 0.0f;
    for (int micro = 0; micro < config_.grad_accum; micro++) {
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

        loss->backward();
        loss_sum += loss->getData().getValue(0, 0);
        loss->release_graph();
    }
    if (config_.grad_accum > 1) {
        optimizer_->scale_grads(1.0f / config_.grad_accum);
    }

    float loss_val = loss_sum / config_.grad_accum;

    // Pre-clip gradient norm, every step: one linear pass over the
    // parameters (<1% of step time) buys the dashboard its gradient-norm
    // track and clip-rate statistic.
    auto params = model_.getAllParameters();
    float grad_norm = utils::compute_grad_norm(params);
    if (step % 100 == 0) {
        metrics_->record_step(step, loss_val, grad_norm);
    }

    optimizer_->clip_grad_norm(5.0f);
    optimizer_->step();

    if (mlog_) {
        auto step_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - step_start).count();
        mlog_->log_step(step, loss_val, optimizer_->current_lr(), grad_norm,
                        step_ms, static_cast<long>(utils::get_memory_mb()));
    }

    if (step % 10 == 0) {
        metrics_->print_progress(step, loss_val);
    }
}

bool Trainer::save_checkpoint(const std::string& path) {
    const std::string tmp = path + ".tmp";
    if (!model_.save(tmp, /*quiet=*/true)) {
        std::cerr << "Warning: failed to write checkpoint " << path << std::endl;
        std::remove(tmp.c_str());
        return false;
    }
    if (!replace_file(tmp, path)) return false;
    std::cout << "Model saved successfully to: " << path << std::endl;
    return true;
}

} // namespace training
