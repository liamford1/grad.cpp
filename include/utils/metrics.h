#pragma once

#include <chrono>
#include <string>

namespace utils {

class TrainingMetrics {
public:
    TrainingMetrics(int total_steps);

    // start_step > 0 marks a resumed run: speed and ETA are computed from
    // steps done this session, not from step numbers (a run resumed at
    // step 20000 has not done 20000 steps in 10 seconds).
    void start_training(int start_step = 0);
    void record_step(int step, float loss, float grad_norm);
    void print_progress(int step, float loss);
    void print_summary();

private:
    int total_steps_;
    int start_step_;
    std::chrono::high_resolution_clock::time_point start_time_;
    float running_loss_;
    int step_count_;
};

void print_header(const std::string& title);
void print_section(const std::string& title);

} // namespace utils
