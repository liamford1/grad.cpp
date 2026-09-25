#include "grad/utils/metrics.h"
#include "grad/utils/training_utils.h"
#include <cstdio>
#include <iostream>
#include <iomanip>

namespace grad::utils {

namespace {
std::string fixed_wall(double s) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f", s);
    return buf;
}
}  // namespace

MetricsLog::MetricsLog(const std::string& path, bool append,
                       int total_steps, long tokens_per_step,
                       long param_count, const std::string& model_desc)
    : out_(path, append ? std::ios::app : std::ios::trunc),
      start_(std::chrono::steady_clock::now()) {
    // The meta row repeats on resume; readers take the last one.
    out_ << "m," << total_steps << "," << tokens_per_step << ","
         << param_count << "," << model_desc << "\n";
    maybe_flush(true);
}

double MetricsLog::wall_s() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
}

void MetricsLog::log_step(int step, float loss, float lr, float grad_norm,
                          long step_ms, long mem_mb) {
    out_ << "t," << step << "," << loss << "," << lr << "," << grad_norm
         << "," << step_ms << "," << mem_mb << "," << fixed_wall(wall_s()) << "\n";
    maybe_flush();
}

void MetricsLog::log_eval(int step, float val_loss) {
    out_ << "e," << step << "," << val_loss << "," << fixed_wall(wall_s()) << "\n";
    maybe_flush(true);
}

void MetricsLog::maybe_flush(bool force) {
    // Flush every few rows so a live dashboard lags at most a few steps
    // and a crash loses almost nothing, without a syscall per step.
    if (force || ++since_flush_ >= 5) {
        out_.flush();
        since_flush_ = 0;
    }
}

TrainingMetrics::TrainingMetrics(int total_steps)
    : total_steps_(total_steps), start_step_(0), running_loss_(0.0), step_count_(0) {}

void TrainingMetrics::start_training(int start_step) {
    start_step_ = start_step;
    start_time_ = std::chrono::steady_clock::now();
}

void TrainingMetrics::record_step(int step, float loss, float grad_norm) {
    running_loss_ += loss;
    step_count_++;

    if (step % 100 == 0) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();

        std::cout << std::setw(10) << step
                  << std::setw(15) << std::fixed << std::setprecision(6) << loss
                  << std::setw(15) << std::fixed << std::setprecision(4) << grad_norm
                  << "  (" << elapsed << "s)"
                  << " [" << get_memory_mb() << "MB]"
                  << std::endl;
    }
}

void TrainingMetrics::print_progress(int step, float loss) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
    float progress = 100.0f * static_cast<float>(step + 1) / static_cast<float>(total_steps_);
    float steps_per_sec = static_cast<float>(step + 1 - start_step_) / static_cast<float>(elapsed + 1);
    int eta_sec = static_cast<int>(static_cast<float>(total_steps_ - step - 1) / (steps_per_sec + 0.0001f));

    std::cout << "\r[" << (step + 1) << "/" << total_steps_ << "] "
              << std::fixed << std::setprecision(1) << progress << "% "
              << "loss=" << std::setprecision(4) << loss << " "
              << "speed=" << std::setprecision(2) << steps_per_sec << "it/s "
              << "eta=" << (eta_sec / 60) << "m" << (eta_sec % 60) << "s     " << std::flush;
}

void TrainingMetrics::print_summary() {
    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - start_time_).count();

    // Everything here covers this process only. A resumed run's earlier
    // segments live in the metrics CSV (`grad watch` sums them).
    std::cout << "\n\n=== Training Complete ===" << std::endl;
    if (start_step_ > 0) {
        std::cout << "Time this session: " << std::fixed << std::setprecision(0) << seconds
                  << "s (steps " << start_step_ << "-" << total_steps_ - 1
                  << "; earlier sessions not included)" << std::endl;
    } else {
        std::cout << "Total time: " << std::fixed << std::setprecision(0) << seconds << "s"
                  << std::endl;
    }
    if (step_count_ > 0) {
        std::cout << "Average loss: " << std::fixed << std::setprecision(4)
                  << running_loss_ / step_count_ << " over " << step_count_ << " steps"
                  << std::endl;
    }
    if (seconds > 0.0 && step_count_ > 0) {
        std::cout << "Speed: " << std::fixed << std::setprecision(2)
                  << step_count_ / seconds << " steps/s" << std::endl;
    }
    std::cout << std::defaultfloat;
}

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
}

void print_section(const std::string& title) {
    std::cout << "\n=== " << title << " ===" << std::endl;
}

}  // namespace grad::utils
