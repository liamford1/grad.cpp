#pragma once
#include "variable.h"
#include "tensor.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <cmath>

class Optimizer {
    public:
        virtual ~Optimizer() = default;
        virtual void step() = 0;
        virtual void zero_grad() = 0;
};

class SGDOptimizer : public Optimizer {
    private:
        float learning_rate;
        std::vector<std::shared_ptr<Variable>> parameters;
    public:
        explicit SGDOptimizer(float lr) : learning_rate(lr) {}

        void add_parameter(const std::shared_ptr<Variable>& param) {
            parameters.push_back(param);
        }

        void step() override {
            for (auto& param : parameters) {
                if (!param->requiresGrad()) continue;

                Tensor& data = param->getData();
                Tensor& grad = param->getGrad();

                int n = data.numel();  
                float* dptr = data.raw();
                float* gptr = grad.raw();

                for (int i = 0; i < n; i++) {
                    dptr[i] -= learning_rate * gptr[i];
                }
            }
        }

        void zero_grad() override {
            for (auto& param : parameters) {
                Tensor& grad = param->getGrad();
                int n = grad.numel();
                float* gptr = grad.raw();
                for (int i = 0; i < n; i++) {
                    gptr[i] = 0.0f;
                }
            }
        }
};

// AdamW: weight decay is applied directly to the weights (decoupled from
// the adaptive gradient scaling), not folded into the gradient as L2. Only
// weight matrices decay - biases and norm parameters (1-row tensors) are
// exempt, following standard practice.
class AdamOptimizer : public Optimizer {
    private:
        std::vector<std::shared_ptr<Variable>> parameters_;
        float lr_;
        float base_lr_;
        float min_lr_;
        float beta1_;
        float beta2_;
        float epsilon_;
        float weight_decay_;
        int step_count_;
        int warmup_steps_;
        int total_steps_;

        std::unordered_map<Variable*, Tensor> m_;
        std::unordered_map<Variable*, Tensor> v_;

        float scheduled_lr() const;
    public:
        AdamOptimizer(const std::vector<std::shared_ptr<Variable>>& parameters, float lr = 3e-4, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8, float weight_decay = 0.01);

        void step() override;
        void zero_grad() override;
        void clip_grad_norm(float max_norm);
        void set_warmup_steps(int steps) { warmup_steps_ = steps; }

        // Linear warmup to base lr, then cosine decay to min_lr at
        // total_steps. Without this, lr stays at base after warmup.
        void set_schedule(int warmup_steps, int total_steps, float min_lr) {
            warmup_steps_ = warmup_steps;
            total_steps_ = total_steps;
            min_lr_ = min_lr;
        }

        float current_lr() const { return lr_; }
};