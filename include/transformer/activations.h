#pragma once

#include "tensor.h"

Tensor gelu(const Tensor& input);
Tensor dropout(const Tensor& input, float dropout_rate, bool training);