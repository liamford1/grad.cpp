#pragma once

#include <cstddef>

// GPU kernels of the resident stream (docs/design/metal-resident.md). Each
// call encodes work and returns without waiting for it; the stream runs
// calls in the order they were made, so a kernel sees every earlier
// kernel's writes. Results become visible to the CPU through the fenced
// Tensor accessors.
//
// Every pointer must come from Tensor::device_data() of a tensor allocated
// in Metal mode, or point inside one (a head's column slice, say); anything
// else throws std::logic_error before encoding. Extents are element counts.
// Without the Metal backend these are stubs that throw std::logic_error;
// they are unreachable, since metal_mode() cannot become true there.
namespace grad::metal::ops {

// y[i] = value
void fill(float* y, size_t n, float value);
// y[i] = x[i]
void copy(const float* x, float* y, size_t n);

// C = alpha * op(A) @ op(B) + beta * C over whole row-major matrices (A is
// (M, K) or, transposed, (K, M); likewise B), the contract of
// blas_sgemm_ex. Runs on MPS. With beta = 0, C is never read.
void gemm(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, bool transA,
          bool transB, float alpha, float beta);

// A batch of GEMMs over strided views: matrix z of the batch starts at
// base + (z / inner) * outer_stride + (z % inner) * inner_stride for each of
// A, B and C, and rows are ld elements apart. Attention's per-(batch, head)
// products are this with z = b * H + h: head h of a (B*S, d) projection is
// a column slice with ld = d, outer stride S*d and inner stride head_size,
// so no head is ever copied.
struct BatchedGemm {
    const float* A = nullptr;
    const float* B = nullptr;
    float* C = nullptr;
    size_t M = 0, N = 0, K = 0;
    bool transA = false, transB = false;
    float alpha = 1.0f, beta = 0.0f;
    size_t lda = 0, ldb = 0, ldc = 0;
    size_t batch = 1;
    size_t inner = 1;
    size_t a_outer = 0, a_inner = 0;
    size_t b_outer = 0, b_inner = 0;
    size_t c_outer = 0, c_inner = 0;
};
void gemm_batched(const BatchedGemm& g);

}  // namespace grad::metal::ops
