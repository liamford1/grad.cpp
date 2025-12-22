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

class AdamOptimizer : public Optimizer {
    private:
        std::vector<std::shared_ptr<Variable>> parameters_;
        float lr_;
        float base_lr_;
        float beta1_;
        float beta2_;
        float epsilon_;
        float weight_decay_;
        int step_count_;
        int warmup_steps_;

        std::unordered_map<Variable*, Tensor> m_;
        std::unordered_map<Variable*, Tensor> v_;
    public:
        AdamOptimizer(const std::vector<std::shared_ptr<Variable>>& parameters, float lr = 3e-4, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8, float weight_decay = 0.01);

        void step() override;
        void zero_grad() override;
        void clip_grad_norm(float max_norm);
        void set_warmup_steps(int steps) { warmup_steps_ = steps; }
};