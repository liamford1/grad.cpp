#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// Metal backend, Apple Silicon only. Two independent uses share one device
// and one command queue:
//
// 1. CPU mode: the threshold GEMM offload. Tensors stay CPU-owned; unified
//    memory means the GPU computes on the same physical pages (MTLBuffer
//    wraps them with no copy - which is why large Tensor storage is
//    page-aligned). The GPU is invoked through exactly one seam,
//    blas_sgemm_ex in blas_wrapper.h, only for matmuls large enough to
//    amortize the dispatch latency, and synchronously: any call can decline
//    before submitting work, and the caller then runs the CPU BLAS path.
//
// 2. Metal mode (grad::set_device, docs/design/metal-resident.md): the
//    resident stream. Tensor storage comes from a pool of shared MTLBuffers
//    (allocate/release below), ops encode kernels into one command stream
//    (metal_ops.h), and the CPU waits only in fence()/synchronize().
//
// Mixed precision (opt-in, CPU-mode offload only): with GRAD_METAL_FP16=1,
// operands are converted to fp16 on the GPU (a compute kernel writes fp16
// copies into persistent private scratch in the same command buffer) and
// the MPS matmul reads them at half the bandwidth; the result matrix stays
// fp32, so alpha/beta accumulation - including beta=1 gradient
// accumulation - is full precision throughout. Off by default: measured as
// a wash at current model scale (BENCHMARKS.md #11).
//
// Environment switches (see grad/utils/env.h; the pre-rename
// TRANSFORMER_* spellings are still honored):
//   GRAD_METAL=0            disable the GPU entirely
//   GRAD_METAL_THRESHOLD=N  min FLOPs (2*M*N*K) to go to GPU (CPU mode)
//   GRAD_METAL_FP16=1       fp16 operands, fp32 accumulate (CPU mode)
//   GRAD_METAL_COMMIT=N     dispatches per committed command buffer in
//                           Metal mode (default 32)
//   GRAD_METAL_MAX_IN_FLIGHT=N  committed command buffers the CPU may run
//                           ahead of the GPU before waiting (default 16)
namespace grad::metal {

// Counters of the resident stream since process start, for benchmarks
// and for tests asserting that a step did not wait.
struct StreamStats {
    uint64_t syncs = 0;            // waits for the GPU (fences that found work, explicit syncs)
    uint64_t command_buffers = 0;  // committed command buffers
    uint64_t dispatches = 0;       // kernels and MPS GEMMs encoded
    uint64_t throttle_waits = 0;   // waits because the CPU ran too far ahead
    size_t live_bytes = 0;         // pool blocks in use by tensors
    size_t peak_live_bytes = 0;
    size_t cached_bytes = 0;  // pool blocks held for reuse (free or awaiting the GPU)
};

#if defined(GRAD_HAS_METAL)
// True if a Metal device is present and not disabled via environment.
bool available();

// True if the GPU path is converting operands to fp16 (affects the
// numerical tolerance the matmul test applies).
bool fp16_active();

// C = alpha * op(A) @ op(B) + beta * C, row-major, same contract as
// blas_sgemm_ex. Returns true once the GPU has computed C. Returns false
// only when nothing was submitted and C is untouched (GPU unavailable,
// pointers not page-aligned, or a Metal object could not be created); the
// caller then runs the CPU path. A command buffer that fails after
// submission may have partially written C, so it throws std::runtime_error
// rather than returning false.
bool sgemm(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, bool transA,
           bool transB, float alpha, float beta);

// True if the resident stream can run here: available(), a GPU of Apple
// family 7 or later (the GEMM kernel uses simdgroup matrices), and the
// kernel library compiled. The first call compiles it.
bool resident_available();
// "ok", or why resident_available() is false.
std::string resident_status();

// Waits for all queued GPU work if any is pending. Tensor accessors call
// this for tensors that have been handed to the GPU; it is a no-op in CPU
// mode. Rethrows, as std::runtime_error, a command buffer failure recorded
// since the last wait.
void fence();
// Commits the open command buffer and waits for everything queued.
void synchronize();
// Commits the open command buffer without waiting.
void flush();

StreamStats stream_stats();

// Resident allocator: page-rounded shared MTLBuffers, cached by size.
// allocate throws std::bad_alloc on failure. release defers the block's
// reuse until the GPU has finished all work queued so far, since queued
// kernels may still read it; it is called by Tensor's deleter.
float* allocate(size_t bytes);
void release(const float* p) noexcept;
// Returns cached free blocks to the system (after waiting for the GPU).
void empty_cache();
#else
inline bool available() {
    return false;
}
inline bool fp16_active() {
    return false;
}
inline bool sgemm(const float*, const float*, float*, size_t, size_t, size_t, bool, bool, float,
                  float) {
    return false;
}
inline bool resident_available() {
    return false;
}
inline std::string resident_status() {
    return "built without the Metal backend";
}
inline void fence() {}
inline void synchronize() {}
inline void flush() {}
inline StreamStats stream_stats() {
    return {};
}
// Unreachable without Metal: metal_mode() never becomes true, so no
// tensor is ever allocated here.
float* allocate(size_t bytes);
void release(const float* p) noexcept;
inline void empty_cache() {}
#endif

}  // namespace grad::metal
