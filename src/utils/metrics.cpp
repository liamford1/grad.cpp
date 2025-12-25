#include "utils/metrics.h"
#include "utils/training_utils.h"
#include <iostream>
#include <iomanip>

namespace utils {

TrainingMetrics::TrainingMetrics(int total_steps)
    : total_steps_(total_steps), running_loss_(0.0f), step_count_(0) {}

void TrainingMetrics::start_training() {
    start_time_ = std::chrono::high_resolution_clock::now();
}

void TrainingMetrics::record_step(int step, float loss, float grad_norm) {
    running_loss_ += loss;
    step_count_++;

    if (step % 100 == 0) {
        auto now = std::chrono::high_resolution_clock::now();
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
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
    float progress = 100.0f * (step + 1) / total_steps_;
    float steps_per_sec = (step + 1) / (float)(elapsed + 1);
    int eta_sec = (int)((total_steps_ - step - 1) / (steps_per_sec + 0.0001f));

    std::cout << "\r[" << (step + 1) << "/" << total_steps_ << "] "
              << std::fixed << std::setprecision(1) << progress << "% "
              << "loss=" << std::setprecision(4) << loss << " "
              << "speed=" << std::setprecision(2) << steps_per_sec << "it/s "
              << "eta=" << (eta_sec / 60) << "m" << (eta_sec % 60) << "s     " << std::flush;
}

void TrainingMetrics::print_summary() {
    auto end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end - start_time_).count();
    float avg_loss = running_loss_ / step_count_;

    std::cout << "\n\n=== Training Complete ===" << std::endl;
    std::cout << "Total time: " << total_time << "s" << std::endl;
    std::cout << "Average loss: " << std::fixed << std::setprecision(4) << avg_loss << std::endl;
    std::cout << "Speed: " << std::fixed << std::setprecision(2)
              << (total_steps_ / (float)total_time) << " steps/s" << std::endl;
}

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
}

void print_section(const std::string& title) {
    std::cout << "\n=== " << title << " ===" << std::endl;
}

} // namespace utils
