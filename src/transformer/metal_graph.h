#pragma once

// Metal-mode versions of the Variable ops (docs/design/metal-resident.md).
// Each builds the node the CPU op builds, with the same children in the
// same order, and its backward closure encodes GPU work where the CPU
// closure runs CPU work; graph traversal, accumulation order and node
// retirement are shared with the CPU path. Variable's ops call these when
// metal_mode() is set. The fused module ops (attention, norms, embeddings,
// the logits projection) keep their Metal versions beside their CPU code.
//
// Library-internal: not installed.

#include "grad/transformer/variable.h"

#include <memory>

namespace grad::metal_graph {

using VarPtr = std::shared_ptr<Variable>;

[[nodiscard]] VarPtr matmul(const VarPtr& a, const VarPtr& b);
[[nodiscard]] VarPtr add(const VarPtr& a, const VarPtr& b);
[[nodiscard]] VarPtr scale(const VarPtr& a, float factor);
[[nodiscard]] VarPtr softmax(const VarPtr& a);
[[nodiscard]] VarPtr gelu(const VarPtr& a);
[[nodiscard]] VarPtr silu(const VarPtr& a);
[[nodiscard]] VarPtr mul(const VarPtr& a, const VarPtr& b);
// Only called when dropout is active (training, rate > 0).
[[nodiscard]] VarPtr dropout(const VarPtr& a, float rate);
[[nodiscard]] VarPtr log_softmax(const VarPtr& a);
[[nodiscard]] VarPtr nll_loss(const VarPtr& a, const VarPtr& targets);
[[nodiscard]] VarPtr cross_entropy(const VarPtr& logits, const VarPtr& targets);

// The gradient of v, allocated (zeroed) if this is its first write, as a
// pointer for encoding.
[[nodiscard]] inline float* grad_for_write(Variable& v) {
    v.ensureGrad();
    return v.getGrad().device_data();
}

}  // namespace grad::metal_graph
