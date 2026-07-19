#pragma once

#include <chrono>
#include <fstream>
#include <string>

namespace utils {

// Append-only per-step training log, one CSV row per step plus eval and
// meta rows. This is the data source for `transformer watch` (the live
// terminal dashboard) and survives crashes/resumes: a resumed run appends
// to the existing file so the dashboard sees the whole history.
//
// Rows:  m,<total_steps>,<tokens_per_step>
//        t,<step>,<loss>,<lr>,<grad_norm|empty>,<step_ms>,<mem_mb>
//        e,<step>,<val_loss>
class MetricsLog {
public:
    MetricsLog(const std::string& path, bool append,
               int total_steps, long tokens_per_step);

    void log_step(int step, float loss, float lr,
                  float grad_norm, bool has_grad_norm,
                  long step_ms, long mem_mb);
    void log_eval(int step, float val_loss);

private:
    std::ofstream out_;
    int since_flush_ = 0;
    void maybe_flush(bool force = false);
};

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
