#pragma once

// Metal (MPS) matmul backend, Apple Silicon only.
//
// Design: tensors never move. They stay CPU-owned; unified memory means the
// GPU computes on the same physical pages (MTLBuffer wraps them with no
// copy - which is why Tensor storage is page-aligned), so there are no
// device copies to keep coherent. The GPU is invoked through exactly one
// seam, blas_sgemm_ex in blas_wrapper.h, and only for matmuls large enough
// to amortize the dispatch latency. Any call can decline before submitting
// work, and the caller then runs the CPU BLAS path instead.
//
// Mixed precision (opt-in): with TRANSFORMER_METAL_FP16=1, operands are
// converted to fp16 on the GPU (a compute kernel writes fp16 copies into
// persistent private scratch in the same command buffer) and the MPS
// matmul reads them at half the bandwidth; the result matrix stays fp32,
// so alpha/beta accumulation - including beta=1 gradient accumulation -
// is full precision throughout. Weights and every tensor the CPU sees
// remain fp32. Off by default: measured as a wash at current model
// scale (BENCHMARKS.md #11).
//
// Environment switches:
//   TRANSFORMER_METAL=0            disable the GPU entirely
//   TRANSFORMER_METAL_THRESHOLD=N  min FLOPs (2*M*N*K) to go to GPU
//   TRANSFORMER_METAL_FP16=1       fp16 operands, fp32 accumulate
namespace metalgpu {

#if defined(__APPLE__)
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
bool sgemm(const float* A, const float* B, float* C,
           int M, int N, int K, bool transA, bool transB,
           float alpha, float beta);
#else
inline bool available() { return false; }
inline bool fp16_active() { return false; }
inline bool sgemm(const float*, const float*, float*,
                  int, int, int, bool, bool, float, float) { return false; }
#endif

}  // namespace metalgpu
