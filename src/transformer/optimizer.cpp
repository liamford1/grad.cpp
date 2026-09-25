#include "grad/transformer/optimizer.h"
#include "grad/transformer/blas_wrapper.h"
#include "grad/transformer/parallel.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <istream>
#include <numbers>
#include <ostream>
#include <vector>

namespace grad {

AdamOptimizer::AdamOptimizer(const std::vector<std::shared_ptr<Variable>>& parameters, float lr,
                             float beta1, float beta2, float epsilon, float weight_decay)
    : parameters_(parameters),
      lr_(lr),
      base_lr_(lr),
      min_lr_(lr),
      beta1_(beta1),
      beta2_(beta2),
      epsilon_(epsilon),
      weight_decay_(weight_decay) {}

float AdamOptimizer::scheduled_lr() const {
    if (warmup_steps_ > 0 && step_count_ <= warmup_steps_) {
        return base_lr_ * (static_cast<float>(step_count_) / static_cast<float>(warmup_steps_));
    }
    if (total_steps_ > warmup_steps_ && step_count_ < total_steps_) {
        // Cosine decay from base_lr to min_lr over the post-warmup steps.
        float progress = static_cast<float>(step_count_ - warmup_steps_)
                         / static_cast<float>(total_steps_ - warmup_steps_);
        float cosine = 0.5f * (1.0f + std::cos(std::numbers::pi_v<float> * progress));
        return min_lr_ + (base_lr_ - min_lr_) * cosine;
    }
    return total_steps_ > 0 ? min_lr_ : base_lr_;
}

void AdamOptimizer::step() {
    step_count_++;
    lr_ = scheduled_lr();

    // pow(float, int) returns double, so the subtraction runs in double
    // and narrows once, here.
    const float bc1 = static_cast<float>(1.0f - std::pow(beta1_, step_count_));
    const float bc2 = static_cast<float>(1.0f - std::pow(beta2_, step_count_));
    const float inv_bc1 = 1.0f / bc1;
    const float inv_bc2 = 1.0f / bc2;

    const float lr = lr_;
    const float eps = epsilon_;
    const float b1 = beta1_;
    const float b2 = beta2_;

    for (auto& param : parameters_) {
        // Empty grad = no gradient reached this parameter this step (grads
        // are lazily allocated), so there is nothing to apply.
        if (!param->requiresGrad() || !param->hasGrad()) continue;

        Tensor& data = param->getData();
        Tensor& grad = param->getGrad();
        Variable* param_ptr = param.get();

        if (m_.find(param_ptr) == m_.end()) {
            m_[param_ptr] = Tensor::zeros_like(data);
            v_[param_ptr] = Tensor::zeros_like(data);
        }

        Tensor& m = m_[param_ptr];
        Tensor& v = v_[param_ptr];

        // Decay only weight matrices; biases and layernorm gamma/beta are
        // (1, N) tensors and are exempt.
        const bool decay_param = data.getIs3D() || data.getRows() > 1;
        const float wd = decay_param ? weight_decay_ : 0.0f;

        const size_t n = data.numel();
        float* dptr = data.raw();
        float* gptr = grad.raw();
        float* mptr = m.raw();
        float* vptr = v.raw();

        parallel_for(n, 65536, [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                float g = gptr[i];

                float mi = mptr[i] = b1 * mptr[i] + (1.0f - b1) * g;
                float vi = vptr[i] = b2 * vptr[i] + (1.0f - b2) * (g * g);

                float m_hat = mi * inv_bc1;
                float v_hat = vi * inv_bc2;

                // AdamW: decoupled decay, applied to the weight itself
                // rather than added to the gradient where Adam's
                // normalization would scale it unevenly across parameters.
                dptr[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + wd * dptr[i]);
            }
        });
    }
}

void AdamOptimizer::zero_grad() {
    for (auto& param : parameters_) {
        Tensor& grad = param->getGrad();
        if (grad.numel() == 0) continue;
        std::memset(grad.raw(), 0, grad.numel() * sizeof(float));
    }
}

void AdamOptimizer::scale_grads(float s) {
    for (auto& param : parameters_) {
        if (!param->requiresGrad() || !param->hasGrad()) continue;
        Tensor& grad = param->getGrad();
        vec_scale_inplace(grad.raw(), s, grad.numel());
    }
}

bool AdamOptimizer::save_state(std::ostream& out) const {
    int32_t step_count = step_count_;
    uint32_t n_params = static_cast<uint32_t>(parameters_.size());
    out.write(reinterpret_cast<const char*>(&step_count), sizeof(step_count));
    out.write(reinterpret_cast<const char*>(&n_params), sizeof(n_params));

    for (const auto& param : parameters_) {
        uint64_t numel = param->getData().numel();
        out.write(reinterpret_cast<const char*>(&numel), sizeof(numel));
        const auto bytes = static_cast<std::streamsize>(numel * sizeof(float));

        auto m_it = m_.find(param.get());
        if (m_it != m_.end()) {
            const Tensor& v = v_.at(param.get());
            out.write(reinterpret_cast<const char*>(m_it->second.raw()), bytes);
            out.write(reinterpret_cast<const char*>(v.raw()), bytes);
        } else {
            // Saving before the first step: moments are implicitly zero.
            std::vector<float> zeros(numel, 0.0f);
            out.write(reinterpret_cast<const char*>(zeros.data()), bytes);
            out.write(reinterpret_cast<const char*>(zeros.data()), bytes);
        }
    }
    return out.good();
}

bool AdamOptimizer::load_state(std::istream& in) {
    int32_t step_count;
    uint32_t n_params;
    in.read(reinterpret_cast<char*>(&step_count), sizeof(step_count));
    in.read(reinterpret_cast<char*>(&n_params), sizeof(n_params));
    if (!in.good() || n_params != parameters_.size()) return false;

    for (const auto& param : parameters_) {
        const Tensor& data = param->getData();
        uint64_t numel;
        in.read(reinterpret_cast<char*>(&numel), sizeof(numel));
        if (!in.good() || numel != static_cast<uint64_t>(data.numel())) return false;

        Variable* key = param.get();
        if (m_.find(key) == m_.end()) {
            m_[key] = Tensor::zeros_like(data);
            v_[key] = Tensor::zeros_like(data);
        }
        const auto bytes = static_cast<std::streamsize>(numel * sizeof(float));
        in.read(reinterpret_cast<char*>(m_[key].raw()), bytes);
        in.read(reinterpret_cast<char*>(v_[key].raw()), bytes);
        if (!in.good()) return false;
    }
    step_count_ = step_count;
    lr_ = scheduled_lr();
    return true;
}

void AdamOptimizer::clip_grad_norm(float max_norm) {
    float total_norm = 0.0f;
    for (auto& param : parameters_) {
        if (!param->requiresGrad() || !param->hasGrad()) continue;
        Tensor& grad = param->getGrad();
        total_norm += vec_sum_squares(grad.raw(), grad.numel());
    }
    total_norm = std::sqrt(total_norm);

    if (total_norm > max_norm) {
        float clip_coef = max_norm / (total_norm + 1e-6f);
        for (auto& param : parameters_) {
            if (!param->requiresGrad() || !param->hasGrad()) continue;
            Tensor& grad = param->getGrad();
            vec_scale_inplace(grad.raw(), clip_coef, grad.numel());
        }
    }
}

}  // namespace grad
