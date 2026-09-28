#pragma once

#include <memory>
#include <span>
#include <string_view>
#include <typeinfo>

// Per-op non-finite check, a debugging aid for long runs
// (docs/design/metal-resident.md, Correctness). With GRAD_METAL_CHECK=1
// every op that records a graph node waits for the GPU after its forward
// and after its backward, and checks what it wrote: its output, then the
// gradients it accumulated into its inputs. The optimizer checks the clip
// norm, the clipped gradients, the weights and both moments. The first
// non-finite value throws std::runtime_error after a report on stderr: the
// op, the phase, the step, the tensor and its shape, the first bad index
// and value, and whether the op's inputs were finite (if they were, this
// op produced the value).
//
//   GRAD_METAL_CHECK=1         enable (any device; Metal is where it pays)
//   GRAD_METAL_CHECK_FROM=N    start checking at optimizer step N
//   GRAD_METAL_CHECK_DUMP=DIR  on failure, write the op's tensors to DIR as
//                              raw little-endian float32, with a manifest
//
// Ops run under NoGradGuard (evaluation) record no node and are not
// checked. Off, each hook is a load of a flag that never changes and a
// not-taken branch; nothing waits and nothing is allocated.
namespace grad {

class Tensor;
class Variable;

namespace op_check {

namespace detail {
[[nodiscard]] bool read_env();
}  // namespace detail

[[nodiscard]] inline bool enabled() {
    static const bool on = detail::read_env();
    return on;
}

// The optimizer step being run, for reports and GRAD_METAL_CHECK_FROM.
void set_step(int step);

// After an op has produced out; op is its backward closure's type, which
// names the function that built it.
void forward(const Variable& out, std::span<const std::shared_ptr<Variable>> inputs,
             const std::type_info& op);
// Around an op's backward closure: before_backward keeps copies of the
// input gradients when a dump directory is set, so a dump holds what the
// closure accumulated into.
void before_backward(std::span<const std::shared_ptr<Variable>> inputs);
void backward(const Variable& out, std::span<const std::shared_ptr<Variable>> inputs,
              const std::type_info& op);

// One tensor outside the graph (optimizer state), named by what.
void tensor(std::string_view what, const Tensor& t);

}  // namespace op_check
}  // namespace grad
