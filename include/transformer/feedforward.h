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
};

#endif