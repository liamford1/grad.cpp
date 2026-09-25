#include "grad/transformer/tensor.h"
#include "grad/transformer/linear.h"
#include "grad/transformer/activations.h"
#include "grad/transformer/feedforward.h"

FeedForward::FeedForward(int d_model, int hidden_dim, float dropout_rate, bool gated) :
    layer1(d_model, resolve_hidden(d_model, hidden_dim, gated), !gated),
    layer2(resolve_hidden(d_model, hidden_dim, gated), d_model, !gated),
    gated_(gated),
    dropout_rate(dropout_rate) {
    if (gated) {
        gate_ = std::make_unique<Linear>(
            d_model, resolve_hidden(d_model, hidden_dim, gated), /*use_bias=*/false);
    }
}

std::shared_ptr<Variable> FeedForward::forward(std::shared_ptr<Variable> input, bool training) const {
    if (gated_) {
        auto gate = gate_->forward(input)->silu();
        auto hidden = gate->mul(layer1.forward(input));
        hidden = hidden->dropout(dropout_rate, training);
        auto output = layer2.forward(hidden);
        return output->dropout(dropout_rate, training);
    }
    auto output = layer1.forward(input);
    output = output->gelu();
    output = output->dropout(dropout_rate, training);
    output = layer2.forward(output);
    return output->dropout(dropout_rate, training);
}
