#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#include "tensor.h"
#include "linear.h"

class FeedForward {
    private:
        Linear layer1;
        Linear layer2;
        float dropout_rate;
    public:
        FeedForward(int d_model, int hidden_dim = -1, float dropout_rate = 0.1f);
        Tensor forward(const Tensor& input, bool training = false) const;

        const Linear& getLayer1() const { return layer1; }
        const Linear& getLayer2() const { return layer2; }

        const Tensor& getLayer1Weights() const { return layer1.getWeights(); }
        const Tensor& getLayer1Bias() const { return layer1.getBias(); }
        const Tensor& getLayer2Weights() const { return layer2.getWeights(); }
        const Tensor& getLayer2Bias() const { return layer2.getBias(); }

        void setWeights(const Tensor& layer1_weights, const Tensor& layer1_bias, const Tensor& layer2_weights, const Tensor& layer2_bias) {
            layer1.setWeights(layer1_weights, layer1_bias);
            layer2.setWeights(layer2_weights, layer2_bias);
        }
};

#endif