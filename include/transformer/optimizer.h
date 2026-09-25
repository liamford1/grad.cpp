#pragma once
#include "variable.h"
#include "tensor.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <cmath>
#include <iosfwd>

class Optimizer {
    public:
        virtual ~Optimizer() = default;
        virtual void step() = 0;
        virtual void zero_grad() = 0;
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
        explicit AdamOptimizer(const std::vector<std::shared_ptr<Variable>>& parameters, float lr = 3e-4, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8, float weight_decay = 0.01);

        void step() override;
        void zero_grad() override;
        void clip_grad_norm(float max_norm);
        void set_warmup_steps(int steps) { warmup_steps_ = steps; }

        // Multiply every parameter gradient by s. Gradient accumulation
        // scales the summed micro-batch gradients down to their mean here,
        // once, after all backward passes. Scaling each micro-batch loss
        // by s before backward gives the same gradients (the loss ops
        // propagate their upstream gradient), but costs a scale per
        // micro-batch and leaves the loss nodes holding scaled values.
        void scale_grads(float s);

        // Serialize / restore Adam state (step count and per-parameter
        // m/v moments) for checkpoint resume. Parameters are matched by
        // position, which is deterministic because getAllParameters()
        // always walks the model in construction order. load_state
        // returns false on any mismatch (count or shape) rather than
        // resuming with misassigned moments.
        [[nodiscard]] bool save_state(std::ostream& out) const;
        [[nodiscard]] bool load_state(std::istream& in);
        int step_count() const { return step_count_; }

        // Linear warmup to base lr, then cosine decay to min_lr at
        // total_steps. Without this, lr stays at base after warmup.
        void set_schedule(int warmup_steps, int total_steps, float min_lr) {
            warmup_steps_ = warmup_steps;
            total_steps_ = total_steps;
            min_lr_ = min_lr;
        }

        float current_lr() const { return lr_; }
};