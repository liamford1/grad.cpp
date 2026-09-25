#include "grad/utils/training_utils.h"
#include <cmath>

#if defined(__APPLE__) || defined(__linux__)
#include <sys/resource.h>
#endif

#ifdef __APPLE__
#include <mach/mach.h>
#elif defined(__linux__)
#include <fstream>
#include <unistd.h>
#endif

namespace grad::utils {

namespace {
constexpr size_t kBytesPerMiB = size_t{1024} * 1024;
}  // namespace

float compute_grad_norm(const std::vector<std::shared_ptr<Variable>>& params) {
    float grad_norm = 0.0f;
    for (const auto& param : params) {
        const Tensor& grad = param->getGrad();
        for (size_t i = 0; i < grad.numel(); i++) {
            float g = grad.raw()[i];
            grad_norm += g * g;
        }
    }
    return std::sqrt(grad_norm);
}

size_t get_memory_mb() {
#ifdef __APPLE__
    struct task_basic_info info;
    mach_msg_type_number_t size = sizeof(info);
    kern_return_t kerr =
        task_info(mach_task_self(), TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &size);
    return (kerr == KERN_SUCCESS) ? info.resident_size / kBytesPerMiB : 0;
#elif defined(__linux__)
    long rss = 0L;
    std::ifstream statm("/proc/self/statm");
    const long page_bytes = sysconf(_SC_PAGESIZE);
    if (statm >> rss >> rss && rss > 0 && page_bytes > 0) {
        return static_cast<size_t>(rss) * static_cast<size_t>(page_bytes) / kBytesPerMiB;
    }
    return 0;
#else
    return 0;
#endif
}

size_t get_peak_memory_mb() {
#if defined(__APPLE__) || defined(__linux__)
    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#ifdef __APPLE__
    return static_cast<size_t>(usage.ru_maxrss) / kBytesPerMiB;
#else
    return static_cast<size_t>(usage.ru_maxrss) / 1024;
#endif
#else
    return 0;
#endif
}

void reshape_batch_to_2d(const Tensor& batch_input, const Tensor& batch_target, Tensor& input_2d,
                         Tensor& target_2d) {
    const size_t batch_size = batch_input.getBatchSize();
    const size_t seq_len = batch_input.getRows();

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            input_2d.setValue(b * seq_len + s, 0, batch_input.getValue(b, s, 0));
            target_2d.setValue(b * seq_len + s, 0, batch_target.getValue(b, s, 0));
        }
    }
}

}  // namespace grad::utils
