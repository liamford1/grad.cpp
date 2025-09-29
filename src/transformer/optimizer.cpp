#include "transformer/optimizer.h"
#include <cmath>
#include <algorithm>

AdamOptimizer::AdamOptimizer(const std::vector<std::shared_ptr<Variable>>& parameters, float lr, float beta1, float beta2, float epsilon, float weight_decay) : parameters_(parameters), lr_(lr), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), weight_decay_(weight_decay), step_count_(0) {}

void AdamOptimizer::step() {
    step_count_++;
    
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
        
        int n = data.numel();
        float* dptr = data.raw();
        float* gptr = grad.raw();
        float* mptr = m.raw();
        float* vptr = v.raw();
        
        for (int i = 0; i < n; i++) {
            float g = gptr[i];
            
            if (weight_decay_ > 0.0f) {
                g += weight_decay_ * dptr[i];
            }
            
            mptr[i] = beta1_ * mptr[i] + (1.0f - beta1_) * g;
            
            vptr[i] = beta2_ * vptr[i] + (1.0f - beta2_) * (g * g);
            
            float m_hat = mptr[i] / (1.0f - std::pow(beta1_, step_count_));
            float v_hat = vptr[i] / (1.0f - std::pow(beta2_, step_count_));
            
            dptr[i] -= lr_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }
}

void AdamOptimizer::zero_grad() {
    for (auto& param : parameters_) {
        Tensor& grad = param->getGrad();
        int n = grad.numel();
        float* gptr = grad.raw();
        for (int i = 0; i < n; i++) {
            gptr[i] = 0.0f;
        }
    }
}

void AdamOptimizer::clip_grad_norm(float max_norm) {
    float total_norm = 0.0f;
    for (auto& param : parameters_) {
        if (!param->requiresGrad()) continue;
        
        Tensor& grad = param->getGrad();
        int n = grad.numel();
        float* gptr = grad.raw();
        
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
            int n = grad.numel();
            float* gptr = grad.raw();
            
            for (int i = 0; i < n; i++) {
                gptr[i] *= clip_coef;
            }
        }
    }
}