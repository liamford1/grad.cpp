#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "tensor.h"

Tensor gelu(const Tensor& input);
Tensor dropout(const Tensor& input, float dropout_rate, bool training);

#endif