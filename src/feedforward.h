#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#include "tensor.h"
#include "linear.h"

class FeedForward {
    private:
        Linear layer1;
        Linear layer2;
    public:
        FeedForward(int d_model, int hidden_dim = -1);
        Tensor forward(const Tensor& input) const;
};

#endif