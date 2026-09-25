#pragma once

#include "tensor.h"
#include <memory>
#include <functional>
#include <vector>
#include <unordered_set>

// Grad mode: whether ops record the autograd graph. On by default. It is
// thread-local, so a NoGradGuard around evaluation on one thread leaves
// training on another untouched; parallel_for workers never build graph
// nodes, so only the thread issuing the ops matters.
class GradMode {
    public:
        [[nodiscard]] static bool is_enabled() noexcept { return enabled_; }
        static void set_enabled(bool enabled) noexcept { enabled_ = enabled; }
    private:
        static inline thread_local bool enabled_ = true;
};

// Disables grad mode for its lifetime and restores the previous state on
// exit, so guards nest. While disabled, op outputs do not require grad,
// record no children and capture no backward closure or forward caches:
// a forward pass allocates only its activations, each freed as soon as the
// next op has consumed it. Leaves created with requires_grad = true (model
// parameters) are unaffected; backward() on an output made under the
// guard throws, as for any root that does not require grad.
class NoGradGuard {
    public:
        NoGradGuard() noexcept : previous_(GradMode::is_enabled()) { GradMode::set_enabled(false); }
        ~NoGradGuard() { GradMode::set_enabled(previous_); }
        NoGradGuard(const NoGradGuard&) = delete;
        NoGradGuard& operator=(const NoGradGuard&) = delete;
    private:
        bool previous_;
};

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
        // activations, about half of what eager grads would cost, and a node
        // whose grad was never touched signals "no gradient flowed here"
        // (its backward fn returns early). No-op when the Variable does
        // not require grad or the grad already exists.
        void ensureGrad();

        // Ops are non-const: each registers *this as a child of its output
        // and hands the backward closure a mutable handle to it, so the
        // graph edge is a real mutation of this node's ownership, not a
        // read.
        [[nodiscard]] std::shared_ptr<Variable> matmul(std::shared_ptr<Variable> other);
        [[nodiscard]] std::shared_ptr<Variable> add(std::shared_ptr<Variable> other);
        [[nodiscard]] std::shared_ptr<Variable> scale(float factor);
        [[nodiscard]] std::shared_ptr<Variable> softmax();

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

// Whether an op over these inputs must record a graph node: grad mode is
// on and at least one input requires grad. Every op, including the fused
// module ops (attention, LayerNorm, embeddings), decides through this.
template <typename... Inputs>
[[nodiscard]] bool compute_requires_grad(const Inputs&... inputs) {
    return GradMode::is_enabled() && (... || inputs->requiresGrad());
}
