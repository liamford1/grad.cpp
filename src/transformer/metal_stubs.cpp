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
void add(const float*, const float*, float*, size_t) {
    unavailable();
}
void accumulate(float*, const float*, size_t) {
    unavailable();
}
void accumulate_scaled(float*, const float*, float, size_t) {
    unavailable();
}
void mul(const float*, const float*, float*, size_t) {
    unavailable();
}
void mul_accumulate(float*, const float*, const float*, size_t) {
    unavailable();
}
void scale(const float*, float, float*, size_t) {
    unavailable();
}
void scale_by(float*, const float*, size_t) {
    unavailable();
}
void add_rows(const float*, const float*, float*, size_t, size_t, size_t) {
    unavailable();
}
void broadcast_add(const float*, Broadcast, const float*, Broadcast, float*, size_t, size_t,
                   size_t) {
    unavailable();
}
void broadcast_reduce(const float*, size_t, size_t, size_t, float*, size_t, size_t, bool, bool) {
    unavailable();
}
void column_sums(const float*, size_t, size_t, float*) {
    unavailable();
}
void gelu(const float*, float*, size_t) {
    unavailable();
}
void gelu_backward(const float*, const float*, float*, size_t) {
    unavailable();
}
void silu(const float*, float*, size_t) {
    unavailable();
}
void silu_backward(const float*, const float*, float*, size_t) {
    unavailable();
}
void layer_norm(const float*, const float*, const float*, float*, float*, float*, size_t, size_t,
                float, bool) {
    unavailable();
}
void layer_norm_backward(const float*, const float*, const float*, const float*, const float*,
                         float*, float*, float*, size_t, size_t, bool) {
    unavailable();
}
void softmax(const float*, float*, size_t, size_t) {
    unavailable();
}
void softmax_backward(const float*, const float*, float*, size_t, size_t) {
    unavailable();
}
void log_softmax(const float*, float*, size_t, size_t) {
    unavailable();
}
void log_softmax_backward(const float*, const float*, float*, size_t, size_t) {
    unavailable();
}
void attention_softmax(float*, size_t, size_t, float) {
    unavailable();
}
void attention_softmax_backward(const float*, float*, const float*, size_t, size_t, float) {
    unavailable();
}
void nll_loss(const float*, const float*, float*, size_t, size_t) {
    unavailable();
}
void nll_loss_backward(const float*, const float*, float*, size_t, size_t) {
    unavailable();
}
void cross_entropy(const float*, const float*, float*, float*, float*, size_t, size_t) {
    unavailable();
}
void cross_entropy_backward(const float*, const float*, const float*, const float*, float*, size_t,
                            size_t) {
    unavailable();
}
void embedding(const float*, const float*, float*, size_t, size_t, float) {
    unavailable();
}
void embedding_backward(const float*, const float*, const float*, size_t, const float*, float*,
                        size_t, float) {
    unavailable();
}
void rope(float*, size_t, size_t, size_t, size_t, const float*, const float*, bool) {
    unavailable();
}
void dropout(const float*, float*, float*, size_t, size_t, uint64_t, uint64_t, float, float) {
    unavailable();
}
void adamw(float*, const float*, float*, float*, size_t, const AdamW&) {
    unavailable();
}
void grad_norm(std::span<const Span>, float, float*) {
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
