#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"

class Linear {
    private:
        int input_dim;
        int output_dim;
        Tensor weights;
        Tensor bias;
        bool use_bias;
    public:
        Linear(int input_dim, int output_dim, bool use_bias = true);
        ~Linear();

        Tensor forward(const Tensor& input) const;
};

#endif