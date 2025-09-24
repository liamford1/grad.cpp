#ifndef LAYER_NORM_H
#define LAYER_NORM_H

#include "tensor.h"

class LayerNorm {
    private:
        int d_model;
        Tensor gamma;
        Tensor beta;
        float epsilon;
    public:
        LayerNorm(int d_model);
        ~LayerNorm();

        Tensor forward(const Tensor& input) const;
};

#endif