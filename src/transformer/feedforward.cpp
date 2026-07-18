#include "transformer/tensor.h"
#include "transformer/linear.h"
#include "transformer/activations.h"
#include "transformer/feedforward.h"
#include <iostream>

FeedForward::FeedForward(int d_model, int hidden_dim, float dropout_rate) :
    layer1(d_model, (hidden_dim == -1) ? 4 * d_model : hidden_dim),
    layer2((hidden_dim == -1) ? 4 * d_model : hidden_dim, d_model),
    dropout_rate(dropout_rate) {}

std::shared_ptr<Variable> FeedForward::forward(std::shared_ptr<Variable> input, bool training) const {
    auto output = layer1.forward(input);
    output = output->gelu();
    output = output->dropout(dropout_rate, training);
    output = layer2.forward(output);
    return output->dropout(dropout_rate, training);
}