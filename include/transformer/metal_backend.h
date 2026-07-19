#pragma once

// Metal (MPS) matmul backend, Apple Silicon only.
//
// Design, informed by this repo's failed CUDA port: tensors never move.
// They stay CPU-owned; unified memory means the GPU computes on the same
// physical pages (MTLBuffer wraps them with no copy - which is why Tensor
// storage is page-aligned). The GPU is invoked through exactly one seam,
// blas_sgemm_ex in blas_wrapper.h, and only for matmuls large enough to
// amortize the dispatch latency. Every call can fall back to the CPU BLAS
// path by returning false, so there is no device state to corrupt.
//
// Environment switches:
//   TRANSFORMER_METAL=0            disable the GPU entirely
//   TRANSFORMER_METAL_THRESHOLD=N  min FLOPs (2*M*N*K) to go to GPU
namespace metalgpu {

#if defined(__APPLE__)
// True if a Metal device is present and not disabled via environment.
bool available();

// C = alpha * op(A) @ op(B) + beta * C, row-major, same contract as
// blas_sgemm_ex. Returns false (leaving C untouched) if the GPU path is
// unavailable or the pointers are not page-aligned; the caller then runs
// the CPU path.
bool sgemm(const float* A, const float* B, float* C,
           int M, int N, int K, bool transA, bool transB,
           float alpha, float beta);
#else
inline bool available() { return false; }
inline bool sgemm(const float*, const float*, float*,
                  int, int, int, bool, bool, float, float) { return false; }
#endif

}  // namespace metalgpu
