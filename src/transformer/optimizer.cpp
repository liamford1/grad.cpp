#include "transformer/optimizer.h"
#include <cmath>
#include <algorithm>

AdamOptimizer::AdamOptimizer(const std::vector<std::shared_ptr<Variable>>& parameters, float lr, float beta1, float beta2, float epsilon, float weight_decay) : parameters_(parameters), lr_(lr), base_lr_(lr), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), weight_decay_(weight_decay), step_count_(0), warmup_steps_(0) {}

void AdamOptimizer::step() {
    step_count_++;

    if (warmup_steps_ > 0 && step_count_ <= warmup_steps_) {
        lr_ = base_lr_ * (static_cast<float>(step_count_) / warmup_steps_);
    } else {
        lr_ = base_lr_;
    }

    const float bc1 = 1.0f - std::pow(beta1_, step_count_);
    const float bc2 = 1.0f - std::pow(beta2_, step_count_);
    const float inv_bc1 = 1.0f / bc1;
    const float inv_bc2 = 1.0f / bc2;

    const float lr  = lr_;
    const float eps = epsilon_;
    const float wd  = weight_decay_;
    const float b1  = beta1_;
    const float b2  = beta2_;
    
    for (auto& param : parameters_) {
        if (!param->requiresGrad()) continue;

        Tensor& data = param->getData();
        Tensor& grad = param->getGrad();
        Variable* param_ptr = param.get();

        Device original_device = data.getDevice();
        Tensor data_cpu = (original_device == Device::CUDA) ? data.to(Device::CPU) : data;
        Tensor grad_cpu = (grad.getDevice() == Device::CUDA) ? grad.to(Device::CPU) : grad;

        if (m_.find(param_ptr) == m_.end()) {
            if (data_cpu.getIs3D()) {
                m_[param_ptr] = Tensor(data_cpu.getBatchSize(), data_cpu.getRows(), data_cpu.getCols(), Device::CPU);
                v_[param_ptr] = Tensor(data_cpu.getBatchSize(), data_cpu.getRows(), data_cpu.getCols(), Device::CPU);
            } else {
                m_[param_ptr] = Tensor(data_cpu.getRows(), data_cpu.getCols(), Device::CPU);
                v_[param_ptr] = Tensor(data_cpu.getRows(), data_cpu.getCols(), Device::CPU);
            }

            m_[param_ptr].fill(0.0f);
            v_[param_ptr].fill(0.0f);
        }

        Tensor& m = m_[param_ptr];
        Tensor& v = v_[param_ptr];

        int n = data_cpu.numel();
        float* dptr = data_cpu.raw();
        float* gptr = grad_cpu.raw();
        float* mptr = m.raw();
        float* vptr = v.raw();

        for (int i = 0; i < n; ++i) {
            float g = gptr[i];
            if (wd > 0.0f) {
                g += wd * dptr[i];
            }

            float mi = mptr[i] = b1 * mptr[i] + (1.0f - b1) * g;
            float vi = vptr[i] = b2 * vptr[i] + (1.0f - b2) * (g * g);

            float m_hat = mi * inv_bc1;
            float v_hat = vi * inv_bc2;

            dptr[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }

        if (original_device == Device::CUDA) {
            Tensor data_cuda = data_cpu.to(Device::CUDA);
            data = data_cuda;
        } else {
            data = data_cpu;
        }
    }
}

void AdamOptimizer::zero_grad() {
    for (auto& param : parameters_) {
        Tensor& grad = param->getGrad();
        grad.fill(0.0f);
    }
}

void AdamOptimizer::clip_grad_norm(float max_norm) {
    float total_norm = 0.0f;
    for (auto& param : parameters_) {
        if (!param->requiresGrad()) continue;

        Tensor& grad = param->getGrad();
        Tensor grad_cpu = (grad.getDevice() == Device::CUDA) ? grad.to(Device::CPU) : grad;

        int n = grad_cpu.numel();
        const float* gptr = grad_cpu.raw();

        for (int i = 0; i < n; i++) {
            total_norm += gptr[i] * gptr[i];
        }
    }
    total_norm = std::sqrt(total_norm);

    if (total_norm > max_norm) {
        float clip_coef = max_norm / (total_norm + 1e-6f);

        for (auto& param : parameters_) {
            if (!param->requiresGrad()) continue;

            Tensor& grad = param->getGrad();
            Device original_device = grad.getDevice();
            Tensor grad_cpu = (original_device == Device::CUDA) ? grad.to(Device::CPU) : grad;

            int n = grad_cpu.numel();
            float* gptr = grad_cpu.raw();

            for (int i = 0; i < n; i++) {
                gptr[i] *= clip_coef;
            }

            if (original_device == Device::CUDA) {
                Tensor grad_cuda = grad_cpu.to(Device::CUDA);
                grad = grad_cuda;
            } else {
                grad = grad_cpu;
            }
        }
    }
}