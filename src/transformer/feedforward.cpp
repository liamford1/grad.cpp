#include "transformer/tensor.h"
#include "transformer/linear.h"
#include "transformer/activations.h"
#include "transformer/feedforward.h"
#include <iostream>

FeedForward::FeedForward(int d_model, int hidden_dim, float dropout_rate) :
    layer1(d_model, (hidden_dim == -1) ? 4 * d_model : hidden_dim),
    layer2((hidden_dim == -1) ? 4 * d_model : hidden_dim, d_model),
    dropout_rate(dropout_rate) {}

Tensor FeedForward::forward(const Tensor& input, bool training) const {
    Tensor output = layer1.forward(input);
    output = gelu(output);
    output = dropout(output, dropout_rate, training);
    output = layer2.forward(output);
    return dropout(output, dropout_rate, training);
}