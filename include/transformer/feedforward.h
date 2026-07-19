#pragma once
#include "tensor.h"
#include "linear.h"
#include <memory>

// Two variants behind one class:
//   gated = false (GPT-2 style): out = W2(gelu(W1 x + b1)) + b2, hidden 4d.
//   gated = true (SwiGLU):       out = W_down(silu(W_gate x) * W_up x),
//     bias-free, hidden 8d/3 rounded up to a multiple of 64 - the same
//     parameter count as the GPT-2 FFN (3 * d * 8d/3 = 8d^2).
// In gated mode layer1 is W_up and layer2 is W_down, so the existing
// accessors (and the d_ff probe in InferenceSession) stay meaningful.
class FeedForward {
    private:
        Linear layer1;
        Linear layer2;
        std::unique_ptr<Linear> gate_;
        bool gated_;
        float dropout_rate;

        static int resolve_hidden(int d_model, int hidden_dim, bool gated) {
            if (hidden_dim != -1) return hidden_dim;
            if (!gated) return 4 * d_model;
            return ((8 * d_model / 3) + 63) / 64 * 64;
        }
    public:
        FeedForward(int d_model, int hidden_dim = -1, float dropout_rate = 0.1f,
                    bool gated = false);
        std::shared_ptr<Variable> forward(std::shared_ptr<Variable> input, bool training = false) const;

        bool isGated() const { return gated_; }

        const Linear& getLayer1() const { return layer1; }
        const Linear& getLayer2() const { return layer2; }

        std::shared_ptr<Variable> getLayer1Weights() const { return layer1.getWeights(); }
        std::shared_ptr<Variable> getLayer1Bias() const { return layer1.getBias(); }
        std::shared_ptr<Variable> getLayer2Weights() const { return layer2.getWeights(); }
        std::shared_ptr<Variable> getLayer2Bias() const { return layer2.getBias(); }
        std::shared_ptr<Variable> getGateWeights() const {
            return gate_ ? gate_->getWeights() : nullptr;
        }

        void setWeights(std::shared_ptr<Variable> layer1_weights, std::shared_ptr<Variable> layer1_bias, std::shared_ptr<Variable> layer2_weights, std::shared_ptr<Variable> layer2_bias) {
            layer1.setWeights(layer1_weights, layer1_bias);
            layer2.setWeights(layer2_weights, layer2_bias);
        }

        void setGatedWeights(std::shared_ptr<Variable> gate_weights,
                             std::shared_ptr<Variable> up_weights,
                             std::shared_ptr<Variable> down_weights) {
            gate_->setWeights(gate_weights, nullptr);
            layer1.setWeights(up_weights, nullptr);
            layer2.setWeights(down_weights, nullptr);
        }
};
