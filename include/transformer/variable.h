#pragma once

#include "tensor.h"
#include <memory>
#include <functional>
#include <vector>
#include <unordered_set>

// A node in the autograd graph: a value, its (lazily allocated) gradient,
// the nodes it was computed from, and a closure that propagates gradient
// to them. Every op below builds one output node and wires its backward
// closure; backward() walks the graph in reverse topological order.
class Variable : public std::enable_shared_from_this<Variable> {
    // Passkey: constructors are public so std::make_shared can reach them,
    // but they require a token only this class can mint. A Variable can
    // therefore never exist outside a shared_ptr, which is exactly the
    // precondition enable_shared_from_this leaves unchecked.
    struct Private { explicit Private() = default; };

    private:
        Tensor data;
        Tensor grad;
        bool requires_grad;
        std::vector<std::shared_ptr<Variable>> children;
        std::function<void()> backward_fn;

    public:
        Variable(Private, const Tensor& data, bool requires_grad = false);
        // Move overloads: op results transfer into their Variable instead
        // of being deep-copied - a full activation-sized memcpy per op
        // otherwise (measured under _platform_memmove, BENCHMARKS.md #10).
        Variable(Private, Tensor&& data, bool requires_grad = false);
        Variable(Private, int rows, int cols, bool requires_grad = false);
        Variable(Private, int batch_size, int rows, int cols, bool requires_grad = false);

        [[nodiscard]] static std::shared_ptr<Variable> create(const Tensor& data, bool requires_grad = false);
        [[nodiscard]] static std::shared_ptr<Variable> create(Tensor&& data, bool requires_grad = false);
        [[nodiscard]] static std::shared_ptr<Variable> create(int rows, int cols, bool requires_grad = false);
        [[nodiscard]] static std::shared_ptr<Variable> create(int batch_size, int rows, int cols, bool requires_grad = false);

        [[nodiscard]] const Tensor& getData() const noexcept { return data; }
        [[nodiscard]] Tensor& getData() noexcept { return data; }
        [[nodiscard]] const Tensor& getGrad() const noexcept { return grad; }
        [[nodiscard]] Tensor& getGrad() noexcept { return grad; }

        [[nodiscard]] bool hasGrad() const noexcept { return grad.numel() > 0; }
        [[nodiscard]] bool requiresGrad() const noexcept { return requires_grad; }

        // Gradients are allocated lazily: construction leaves grad empty,
        // and backward functions call ensureGrad() (allocate + zero) before
        // their first write. The forward pass therefore holds only
        // activations - half the graph's former footprint - and a node
        // whose grad was never touched signals "no gradient flowed here"
        // (its backward fn returns early). No-op when the Variable does
        // not require grad or the grad already exists.
        void ensureGrad();

        // Ops are non-const: each registers *this as a child of its output
        // and hands the backward closure a mutable handle to it, so the
        // graph edge is a real mutation of this node's ownership, not a
        // read. (An earlier version declared them const and laundered
        // the pointer back through const_pointer_cast, which lied about
        // exactly this.)
        [[nodiscard]] std::shared_ptr<Variable> matmul(std::shared_ptr<Variable> other);
        [[nodiscard]] std::shared_ptr<Variable> add(std::shared_ptr<Variable> other);
        [[nodiscard]] std::shared_ptr<Variable> scale(float factor);
        [[nodiscard]] std::shared_ptr<Variable> softmax();

        [[nodiscard]] std::shared_ptr<Variable> cross_entropy_loss(std::shared_ptr<Variable> targets);
        [[nodiscard]] std::shared_ptr<Variable> gelu();
        [[nodiscard]] std::shared_ptr<Variable> silu();
        // Elementwise product of two same-shape tensors (the SwiGLU gate).
        [[nodiscard]] std::shared_ptr<Variable> mul(std::shared_ptr<Variable> other);
        [[nodiscard]] std::shared_ptr<Variable> dropout(float rate, bool training);

        [[nodiscard]] std::shared_ptr<Variable> log_softmax();
        [[nodiscard]] std::shared_ptr<Variable> nll_loss(std::shared_ptr<Variable> targets);

        void backward();
        void zeroGrad();
        void release_graph();

        void addChild(std::shared_ptr<Variable> child) { children.push_back(std::move(child)); }
        void setBackwardFn(std::function<void()> fn) { backward_fn = std::move(fn); }
    private:
        void topologicalSort(std::vector<std::shared_ptr<Variable>>& sorted, std::unordered_set<Variable*>& visited);
        [[nodiscard]] std::shared_ptr<Variable> createOutput(Tensor&& result, bool needs_grad);
};
