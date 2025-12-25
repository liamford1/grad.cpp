#pragma once

#include <chrono>
#include <string>

namespace utils {

class TrainingMetrics {
public:
    TrainingMetrics(int total_steps);

    void start_training();
    void record_step(int step, float loss, float grad_norm);
    void print_progress(int step, float loss);
    void print_summary();

private:
    int total_steps_;
    std::chrono::high_resolution_clock::time_point start_time_;
    float running_loss_;
    int step_count_;
};

void print_header(const std::string& title);
void print_section(const std::string& title);

} // namespace utils
