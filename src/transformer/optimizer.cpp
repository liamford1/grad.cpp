#include "transformer/optimizer.h"
#include "transformer/blas_wrapper.h"
#include "transformer/parallel.h"
#include <cmath>
#include <cstring>
#include <algorithm>

AdamOptimizer::AdamOptimizer(const std::vector<std::shared_ptr<Variable>>& parameters, float lr, float beta1, float beta2, float epsilon, float weight_decay) : parameters_(parameters), lr_(lr), base_lr_(lr), min_lr_(lr), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), weight_decay_(weight_decay), step_count_(0), warmup_steps_(0), total_steps_(0) {}

float AdamOptimizer::scheduled_lr() const {
    if (warmup_steps_ > 0 && step_count_ <= warmup_steps_) {
        return base_lr_ * (static_cast<float>(step_count_) / warmup_steps_);
    }
    if (total_steps_ > warmup_steps_ && step_count_ < total_steps_) {
        // Cosine decay from base_lr to min_lr over the post-warmup steps.
        float progress = static_cast<float>(step_count_ - warmup_steps_)
                       / (total_steps_ - warmup_steps_);
        float cosine = 0.5f * (1.0f + std::cos(3.14159265f * progress));
        return min_lr_ + (base_lr_ - min_lr_) * cosine;
    }
    return total_steps_ > 0 ? min_lr_ : base_lr_;
}

void AdamOptimizer::step() {
    step_count_++;
    lr_ = scheduled_lr();

    const float bc1 = 1.0f - std::pow(beta1_, step_count_);
    const float bc2 = 1.0f - std::pow(beta2_, step_count_);
    const float inv_bc1 = 1.0f / bc1;
    const float inv_bc2 = 1.0f / bc2;

    const float lr  = lr_;
    const float eps = epsilon_;
    const float b1  = beta1_;
    const float b2  = beta2_;

    for (auto& param : parameters_) {
        if (!param->requiresGrad()) continue;

        Tensor& data = param->getData();
        Tensor& grad = param->getGrad();
        Variable* param_ptr = param.get();

        if (m_.find(param_ptr) == m_.end()) {
            if (data.getIs3D()) {
                m_[param_ptr] = Tensor(data.getBatchSize(), data.getRows(), data.getCols());
                v_[param_ptr] = Tensor(data.getBatchSize(), data.getRows(), data.getCols());
            } else {
                m_[param_ptr] = Tensor(data.getRows(), data.getCols());
                v_[param_ptr] = Tensor(data.getRows(), data.getCols());
            }

            m_[param_ptr].fill(0.0f);
            v_[param_ptr].fill(0.0f);
        }

        Tensor& m = m_[param_ptr];
        Tensor& v = v_[param_ptr];

        // Decay only weight matrices; biases and layernorm gamma/beta are
        // (1, N) tensors and are exempt.
        const bool decay_param = data.getIs3D() || data.getRows() > 1;
        const float wd = decay_param ? weight_decay_ : 0.0f;

        int n = data.numel();
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
        std::memset(grad.raw(), 0, grad.numel() * sizeof(float));
    }
}

void AdamOptimizer::clip_grad_norm(float max_norm) {
    float total_norm = 0.0f;
    for (auto& param : parameters_) {
        if (!param->requiresGrad()) continue;
        Tensor& grad = param->getGrad();
        total_norm += vec_sum_squares(grad.raw(), grad.numel());
    }
    total_norm = std::sqrt(total_norm);

    if (total_norm > max_norm) {
        float clip_coef = max_norm / (total_norm + 1e-6f);
        for (auto& param : parameters_) {
            if (!param->requiresGrad()) continue;
            Tensor& grad = param->getGrad();
            vec_scale_inplace(grad.raw(), clip_coef, grad.numel());
        }
    }
}