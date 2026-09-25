#pragma once
#include "grad/transformer/tensor.h"
#include "grad/transformer/variable.h"
#include <memory>

namespace grad {

// rms = true turns the layer into an RMSNorm: y = x / rms(x) * gamma with
// no mean subtraction and no beta shift (the modern-arch norm). beta is
// still constructed (zeros) so the class shape and checkpoint layout stay
// identical across both modes; it never receives gradients in RMS mode,
// so the lazily-allocating optimizer skips it entirely.
class LayerNorm {
private:
    int d_model_;
    std::shared_ptr<Variable> gamma;
    std::shared_ptr<Variable> beta;
    float epsilon;
    bool rms_;

public:
    explicit LayerNorm(int d_model, bool rms = false);
    std::shared_ptr<Variable> forward(const std::shared_ptr<Variable>& input) const;

    std::shared_ptr<Variable> getGamma() const { return gamma; }
    std::shared_ptr<Variable> getBeta() const { return beta; }
    float getEpsilon() const { return epsilon; }
    bool isRMS() const { return rms_; }

    std::vector<std::shared_ptr<Variable>> parameters() const { return {gamma, beta}; }

    void setParams(const Tensor& new_gamma, const Tensor& new_beta) {
        gamma = Variable::create(new_gamma, true);
        beta = Variable::create(new_beta, true);
    }
};

}  // namespace grad
