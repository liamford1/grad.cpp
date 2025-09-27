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

        const Tensor& getGamma() const { return gamma; }
        const Tensor& getBeta() const { return beta; }

        void setParams(const Tensor& new_gamma, const Tensor& new_beta) {
            gamma = new_gamma;
            beta = new_beta;
        }
};

#endif