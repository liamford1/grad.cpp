#pragma once
#include "tensor.h"
#include "linear.h"

class FeedForward {
    private:
        Linear layer1;
        Linear layer2;
        float dropout_rate;
    public:
        FeedForward(int d_model, int hidden_dim = -1, float dropout_rate = 0.1f);
        std::shared_ptr<Variable> forward(std::shared_ptr<Variable> input, bool training = false) const;

        const Linear& getLayer1() const { return layer1; }
        const Linear& getLayer2() const { return layer2; }

        std::shared_ptr<Variable> getLayer1Weights() const { return layer1.getWeights(); }
        std::shared_ptr<Variable> getLayer1Bias() const { return layer1.getBias(); }
        std::shared_ptr<Variable> getLayer2Weights() const { return layer2.getWeights(); }
        std::shared_ptr<Variable> getLayer2Bias() const { return layer2.getBias(); }

        void setWeights(std::shared_ptr<Variable> layer1_weights, std::shared_ptr<Variable> layer1_bias, std::shared_ptr<Variable> layer2_weights, std::shared_ptr<Variable> layer2_bias) {
            layer1.setWeights(layer1_weights, layer1_bias);
            layer2.setWeights(layer2_weights, layer2_bias);
        }
};