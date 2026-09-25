#pragma once

#include <cstdlib>

// Runtime switches read from the environment:
//   GRAD_THREADS=N           thread pool size (default: hardware threads)
//   GRAD_METAL=0             disable the Metal GPU backend
//   GRAD_METAL_THRESHOLD=N   min FLOPs (2*M*N*K) for a matmul to go to GPU
//   GRAD_METAL_FP16=1        fp16 GPU operands, fp32 accumulate
// Each was previously spelled TRANSFORMER_*; the old name is still read
// when the new one is unset, so existing scripts keep working.
namespace grad::env {

// The value of `name`, else of `legacy` (its pre-rename spelling), else
// nullptr. A variable set to the empty string counts as set.
[[nodiscard]] inline const char* lookup(const char* name, const char* legacy) noexcept {
    if (const char* value = std::getenv(name)) return value;
    return std::getenv(legacy);
}

}  // namespace grad::env
