#ifndef VARIABLE_H
#define VARIABLE_H

#include "tensor.h"
#include <memory>
#include <functional>
#include <vector>
#include <unordered_set>

class Variable : public std::enable_shared_from_this<Variable> {
    private:
        Tensor data;
        Tensor grad;
        bool requires_grad;
        std::vector<std::shared_ptr<Variable>> children;
        std::function<void()> backward_fn;
        mutable bool visited = false;
        
    public:
        Variable(const Tensor& data, bool requires_grad = false);
        Variable(int rows, int cols, bool requires_grad = false);
        Variable(int batch_size, int rows, int cols, bool requires_grad = false);

        static std::shared_ptr<Variable> create(const Tensor& data, bool requires_grad = false);
        static std::shared_ptr<Variable> create(int rows, int cols, bool requires_grad = false);
        static std::shared_ptr<Variable> create(int batch_size, int rows, int cols, bool requires_grad = false);
        
        const Tensor& getData() const { return data; }
        Tensor& getData() { return data; }
        const Tensor& getGrad() const { return grad; }
        Tensor& getGrad() { return grad; }
        
        bool requiresGrad() const { return requires_grad; }
        
        std::shared_ptr<Variable> matmul(std::shared_ptr<Variable> other) const;
        std::shared_ptr<Variable> add(std::shared_ptr<Variable> other) const;
        std::shared_ptr<Variable> scale(float factor) const;
        std::shared_ptr<Variable> softmax() const;
        
        void backward();
        void zeroGrad();

        void addChild(std::shared_ptr<Variable> child) { children.push_back(child); }
        void setBackwardFn(std::function<void()> fn) { backward_fn = fn; } 
    private:
        void topologicalSort(std::vector<std::shared_ptr<Variable>>& sorted, std::unordered_set<Variable*>& visited) const;
        std::shared_ptr<Variable> createOutput(const Tensor& result, bool needs_grad) const;
};

#endif