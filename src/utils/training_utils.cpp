#include "utils/training_utils.h"
#include <cmath>
#include <mach/mach.h>

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
    struct task_basic_info info;
    mach_msg_type_number_t size = sizeof(info);
    kern_return_t kerr = task_info(mach_task_self(),
                                    TASK_BASIC_INFO,
                                    (task_info_t)&info,
                                    &size);
    return (kerr == KERN_SUCCESS) ? info.resident_size / (1024 * 1024) : 0;
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
