#include "transformer/variable.h"
#include <cmath>
#include <iostream>
#include <algorithm>

Variable::Variable(const Tensor& data, bool requires_grad) 
    : data(data), requires_grad(requires_grad) {
    if (requires_grad) {
        if (data.getIs3D()) {
            grad = Tensor(data.getBatchSize(), data.getRows(), data.getCols());
        } else {
            grad = Tensor(data.getRows(), data.getCols());
        }
        grad.fill(0.0f);
    }
}

Variable::Variable(int rows, int cols, bool requires_grad) 
    : data(rows, cols), requires_grad(requires_grad) {
    if (requires_grad) {
        grad = Tensor(rows, cols);
        grad.fill(0.0f);
    }
}

Variable::Variable(int batch_size, int rows, int cols, bool requires_grad) 
    : data(batch_size, rows, cols), requires_grad(requires_grad) {
    if (requires_grad) {
        grad = Tensor(batch_size, rows, cols);
        grad.fill(0.0f);
    }
}

std::shared_ptr<Variable> Variable::create(const Tensor& data, bool requires_grad) {
    return std::make_shared<Variable>(data, requires_grad);
}

std::shared_ptr<Variable> Variable::create(int rows, int cols, bool requires_grad) {
    return std::make_shared<Variable>(rows, cols, requires_grad);
}

std::shared_ptr<Variable> Variable::create(int batch_size, int rows, int cols, bool requires_grad) {
    return std::make_shared<Variable>(batch_size, rows, cols, requires_grad);
}

std::shared_ptr<Variable> Variable::createOutput(const Tensor& result, bool needs_grad) const {
    return std::make_shared<Variable>(result, needs_grad);
}

std::shared_ptr<Variable> Variable::matmul(std::shared_ptr<Variable> other) const {
    Tensor result = this->data.matmul(other->data);
    bool needs_grad = this->requires_grad || other->requires_grad;
    
    auto output = createOutput(result, needs_grad);
    
    if (needs_grad) {
        auto self_ptr = std::const_pointer_cast<Variable>(
            std::static_pointer_cast<const Variable>(shared_from_this())
        );
        
        output->addChild(self_ptr);
        output->addChild(other);
        
        output->setBackwardFn([self_ptr, other, output]() {
            if (self_ptr->requires_grad) {
                Tensor other_transposed = other->data.transpose();
                Tensor self_grad = output->grad.matmul(other_transposed);
                self_ptr->grad = self_ptr->grad.add(self_grad);
            }
            if (other->requires_grad) {
                Tensor self_transposed = self_ptr->data.transpose();
                Tensor other_grad = self_transposed.matmul(output->grad);
                other->grad = other->grad.add(other_grad);
            }
        });
    }
    return output;
}

std::shared_ptr<Variable> Variable::add(std::shared_ptr<Variable> other) const {
    Tensor result = this->data.add(other->data);
    bool needs_grad = this->requires_grad || other->requires_grad;
    auto output = createOutput(result, needs_grad);
    
    if (needs_grad) {
        auto self_ptr = std::const_pointer_cast<Variable>(
            std::static_pointer_cast<const Variable>(shared_from_this())
        );
        
        output->addChild(self_ptr);
        output->addChild(other);
        
        output->setBackwardFn([self_ptr, other, output]() {
            if (self_ptr->requires_grad) {
                self_ptr->grad = self_ptr->grad.add(output->grad);
            }
            if (other->requires_grad) {
                other->grad = other->grad.add(output->grad);
            }
        });
    }
    return output;
}

std::shared_ptr<Variable> Variable::scale(float factor) const {
    Tensor result = this->data.scale(factor);
    auto output = createOutput(result, this->requires_grad);
    
    if (this->requires_grad) {
        auto self_ptr = std::const_pointer_cast<Variable>(
            std::static_pointer_cast<const Variable>(shared_from_this())
        );
        
        output->addChild(self_ptr);
        output->setBackwardFn([self_ptr, factor, output]() {
            if (self_ptr->requires_grad) {
                Tensor scaled_grad = output->grad.scale(factor);
                self_ptr->grad = self_ptr->grad.add(scaled_grad);
            }
        });
    }
    return output;
}

std::shared_ptr<Variable> Variable::softmax() const {
    Tensor result = this->data.softmax();
    auto output = createOutput(result, this->requires_grad);
    
    if (this->requires_grad) {
        auto self_ptr = std::const_pointer_cast<Variable>(
            std::static_pointer_cast<const Variable>(shared_from_this())
        );
        
        output->addChild(self_ptr);
        output->setBackwardFn([self_ptr, result, output]() {
            if (self_ptr->requires_grad) {
                if (result.getIs3D()) {
                    Tensor temp_grad(result.getBatchSize(), result.getRows(), result.getCols());
                    temp_grad.fill(0.0f);
                    
                    for (int b = 0; b < result.getBatchSize(); b++) {
                        for (int i = 0; i < result.getRows(); i++) {
                            float dot_product = 0.0f;
                            for (int j = 0; j < result.getCols(); j++) {
                                dot_product += result.getValue(b, i, j) * output->grad.getValue(b, i, j);
                            }
                            for (int j = 0; j < result.getCols(); j++) {
                                float softmax_grad = result.getValue(b, i, j) * 
                                    (output->grad.getValue(b, i, j) - dot_product);
                                temp_grad.setValue(b, i, j, softmax_grad);
                            }
                        }
                    }
                    self_ptr->grad = self_ptr->grad.add(temp_grad);
                } else {
                    Tensor temp_grad(result.getRows(), result.getCols());
                    temp_grad.fill(0.0f);
                    
                    for (int i = 0; i < result.getRows(); i++) {
                        float dot_product = 0.0f;
                        for (int j = 0; j < result.getCols(); j++) {
                            dot_product += result.getValue(i, j) * output->grad.getValue(i, j);
                        }
                        for (int j = 0; j < result.getCols(); j++) {
                            float softmax_grad = result.getValue(i, j) * 
                                (output->grad.getValue(i, j) - dot_product);
                            temp_grad.setValue(i, j, softmax_grad);
                        }
                    }
                    self_ptr->grad = self_ptr->grad.add(temp_grad);
                }
            }
        });
    }
    return output;
}

std::shared_ptr<Variable> Variable::cross_entropy_loss(std::shared_ptr<Variable> targets) const {
    if (!this->data.getIs3D() && !targets->data.getIs3D()) {
        Tensor loss_tensor(1, 1);
        float total_loss = 0.0f;
        
        if (targets->data.getCols() == 1) { 
            for (int i = 0; i < this->data.getRows(); i++) {
                int target_idx = static_cast<int>(targets->data.getValue(i, 0));
                if (target_idx >= 0 && target_idx < this->data.getCols()) {
                    float prob = std::max(this->data.getValue(i, target_idx), 1e-15f);
                    total_loss -= std::log(prob);
                }
            }
        } else {
            for (int i = 0; i < this->data.getRows(); i++) {
                for (int j = 0; j < this->data.getCols(); j++) {
                    if (targets->data.getValue(i, j) > 0.0f) {
                        float prob = std::max(this->data.getValue(i, j), 1e-15f);
                        total_loss -= targets->data.getValue(i, j) * std::log(prob);
                    }
                }
            }
        }
        total_loss /= this->data.getRows();
        loss_tensor.setValue(0, 0, total_loss);
        auto output = createOutput(loss_tensor, this->requires_grad || targets->requires_grad);
        
        if (output->requires_grad) {
            auto self_ptr = std::const_pointer_cast<Variable>(shared_from_this());

            output->addChild(self_ptr);
            output->addChild(targets);
            output->setBackwardFn([self_ptr, targets, output]() {
                if (self_ptr->requires_grad) {
                    if (targets->data.getCols() == 1) {
                        Tensor grad_tensor(self_ptr->data.getRows(), self_ptr->data.getCols());
                        grad_tensor.fill(0.0f);
                        float scale = 1.0f / self_ptr->data.getRows();
                        
                        for (int i = 0; i < self_ptr->data.getRows(); i++) {
                            int target_idx = static_cast<int>(targets->data.getValue(i, 0));
                            if (target_idx >= 0 && target_idx < self_ptr->data.getCols()) {
                                for (int j = 0; j < self_ptr->data.getCols(); j++) {
                                    float grad_val = self_ptr->data.getValue(i, j) * scale;
                                    if (j == target_idx) {
                                        grad_val -= scale;
                                    }
                                    grad_tensor.setValue(i, j, grad_val);
                                }
                            }
                        }
                        self_ptr->grad = self_ptr->grad.add(grad_tensor);
                    } else {
                        Tensor diff = self_ptr->data.subtract(targets->data);
                        Tensor scaled_diff = diff.scale(1.0f / self_ptr->data.getRows());
                        self_ptr->grad = self_ptr->grad.add(scaled_diff);
                    }
                }
            });
        }
        return output;
    } else {
        throw std::runtime_error("Batched cross-entropy not yet implemented");
    }
}

void Variable::topologicalSort(std::vector<std::shared_ptr<Variable>>& sorted, 
                              std::unordered_set<Variable*>& visited) const {
    if (visited.find(const_cast<Variable*>(this)) != visited.end()) {
        return;
    }
    
    visited.insert(const_cast<Variable*>(this));
    
    for (const auto& child : children) {
        child->topologicalSort(sorted, visited);
    }
    
    sorted.push_back(std::const_pointer_cast<Variable>(
        std::static_pointer_cast<const Variable>(shared_from_this())
    ));
}

void Variable::backward() {
    if (!requires_grad) {
        std::cerr << "Warning: backward() called on Variable that doesn't require grad" << std::endl;
        return;
    }
    
    grad.fill(1.0f);
    
    std::vector<std::shared_ptr<Variable>> sorted;
    std::unordered_set<Variable*> visited;
    topologicalSort(sorted, visited);
    
    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
        if ((*it)->backward_fn) {
            (*it)->backward_fn();
        }
    }
}

void Variable::zeroGrad() {
    if (requires_grad) {
        grad.fill(0.0f);
    }
}