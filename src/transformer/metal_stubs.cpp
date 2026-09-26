// The Metal backend's entry points for builds without it (every non-Apple
// build, or -DGRAD_METAL_BACKEND=OFF). metal_mode() cannot become true
// there, so nothing reaches these; they throw rather than fail silently if
// something ever does.
#include "grad/transformer/metal_backend.h"
#include "grad/transformer/metal_ops.h"

#include <stdexcept>

namespace grad::metal {

namespace {
[[noreturn]] void unavailable() {
    throw std::logic_error("grad was built without the Metal backend");
}
}  // namespace

float* allocate(size_t) {
    unavailable();
}

void release(const float*) noexcept {}

namespace ops {

void fill(float*, size_t, float) {
    unavailable();
}
void copy(const float*, float*, size_t) {
    unavailable();
}
void gemm(const float*, const float*, float*, size_t, size_t, size_t, bool, bool, float, float) {
    unavailable();
}
void gemm_batched(const BatchedGemm&) {
    unavailable();
}

}  // namespace ops
}  // namespace grad::metal
