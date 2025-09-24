#include "transformer/tensor.h"
#include "transformer/linear.h"
#include "transformer/feedforward.h"
#include <iostream>

FeedForward::FeedForward(int d_model, int hidden_dim) :
    layer1(d_model, (hidden_dim == -1) ? 4 * d_model : hidden_dim),
    layer2((hidden_dim == -1) ? 4 * d_model : hidden_dim, d_model) {}

Tensor FeedForward::forward(const Tensor& input) const {
    Tensor output = layer1.forward(input);
    output = output.relu();
    return layer2.forward(output);
}