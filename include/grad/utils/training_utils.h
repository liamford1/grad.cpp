#pragma once

#include "grad/transformer/variable.h"
#include "grad/transformer/tensor.h"
#include <vector>
#include <memory>

namespace grad::utils {

float compute_grad_norm(const std::vector<std::shared_ptr<Variable>>& params);

size_t get_memory_mb();
size_t get_peak_memory_mb();

void reshape_batch_to_2d(const Tensor& batch_input, const Tensor& batch_target,
                         Tensor& input_2d, Tensor& target_2d);

}  // namespace grad::utils
