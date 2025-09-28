#pragma once
#include "variable.h"
#include "tensor.h"
#include <vector>
#include <memory>

class SGDOptimizer {
    private:
        float learning_rate;
        std::vector<std::shared_ptr<Variable>> parameters;
    public:
        explicit SGDOptimizer(float lr) : learning_rate(lr) {}

        void add_parameter(const std::shared_ptr<Variable>& param) {
            parameters.push_back(param);
        }

        void step() {
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

        void zero_grad() {
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