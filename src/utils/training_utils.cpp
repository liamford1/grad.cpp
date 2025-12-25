#include "utils/training_utils.h"
#include <cmath>

#ifdef __APPLE__
    #include <mach/mach.h>
#elif defined(__linux__)
    #include <sys/sysinfo.h>
    #include <fstream>
    #include <unistd.h>
#endif

namespace utils {

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
    kern_return_t kerr = task_info(mach_task_self(),
                                    TASK_BASIC_INFO,
                                    (task_info_t)&info,
                                    &size);
    return (kerr == KERN_SUCCESS) ? info.resident_size / (1024 * 1024) : 0;
#elif defined(__linux__)
    long rss = 0L;
    std::ifstream statm("/proc/self/statm");
    if (statm >> rss >> rss) {
        return (rss * sysconf(_SC_PAGESIZE)) / (1024 * 1024);
    }
    return 0L;
#else
    return 0L;
#endif
}

void reshape_batch_to_2d(const Tensor& batch_input, const Tensor& batch_target,
                         Tensor& input_2d, Tensor& target_2d) {
    int batch_size = batch_input.getBatchSize();
    int seq_len = batch_input.getRows();

    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            input_2d.setValue(b * seq_len + s, 0, batch_input.getValue(b, s, 0));
            target_2d.setValue(b * seq_len + s, 0, batch_target.getValue(b, s, 0));
        }
    }
}

} // namespace utils
