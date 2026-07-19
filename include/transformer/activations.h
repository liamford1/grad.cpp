#pragma once

#include "tensor.h"
#include <cstddef>

Tensor dropout(const Tensor& input, float dropout_rate, bool training);

// Fills mask[i] with 0 (dropped) or scale (kept), keeping each element with
// probability 1 - dropout_rate. Uses a thread-local xorshift generator and
// compares in the integer domain, so it costs a few cycles per element
// instead of a std::mt19937 + uniform_real_distribution draw.
void fill_dropout_mask(float* mask, size_t n, float dropout_rate, float scale);
