#pragma once

// Cross-platform BLAS + vDSP Wrapper
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#include "grad/transformer/metal_backend.h"
#include "grad/utils/env.h"
#include "grad/utils/narrow.h"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>

namespace grad {

// Matmuls above this many FLOPs (2*M*N*K) route to the Metal GPU when one
// is available. The default crossover comes from measuring MPS against
// Accelerate on an M2 Pro across model scales (see BENCHMARKS.md): below
// ~10 GFLOPs the AMX units win or tie (the entire 22M-param config stays
// on CPU - measured, not assumed); above it the GPU pulls ahead, reaching
// 1.5-1.8x at 60-120M-param shapes. Tune with GRAD_METAL_THRESHOLD;
// disable with GRAD_METAL=0.
inline bool metal_worthwhile(size_t flops) {
    static const long long threshold = [] {
        if (const char* value =
                env::lookup("GRAD_METAL_THRESHOLD", "TRANSFORMER_METAL_THRESHOLD")) {
            return std::atoll(value);
        }
        return 10LL * 1000 * 1000 * 1000;
    }();
    return flops >= static_cast<size_t>(threshold);
}

// The integer type the linked CBLAS takes for sizes and strides, read off
// cblas_sdot's signature: int for OpenBLAS and reference CBLAS (LP64),
// long for Accelerate built with ACCELERATE_LAPACK_ILP64.
namespace blas_detail {
template <typename R, typename N, typename... Rest>
N first_param(R (*)(N, Rest...));
}  // namespace blas_detail
using blas_int = decltype(blas_detail::first_param(&cblas_sdot));

// Size convention: extents are size_t everywhere in grad::core and are
// narrowed to the BLAS's integer type only here. The GEMM entry points
// check the narrowing (grad::narrow throws std::overflow_error), once per
// call. The vector helpers are called per row inside kernels, so they
// only assert: every length they see is bounded by some tensor's numel,
// and Tensor caps numel at kMaxTensorElements, below INT_MAX.
inline blas_int vec_length(size_t n) noexcept {
    assert(n <= static_cast<size_t>(std::numeric_limits<int>::max()));
    return static_cast<blas_int>(n);
}

// C = alpha * op(A) @ op(B) + beta * C
// A, B, C are row-major. op(A) is (M,K), op(B) is (K,N), C is (M,N).
// beta = 1 accumulates into C in place - used by backward passes to add
// gradient contributions without materializing a temporary.
//
// Large products try the GPU first. metal::sgemm returns false only
// when it submitted nothing and left C untouched, which is what makes
// rerunning on the CPU with the same beta correct; a GPU failure after
// submission throws instead of coming back here.
inline void blas_sgemm_ex(const float* A, const float* B, float* C, size_t M, size_t N, size_t K,
                          bool transA, bool transB, float alpha, float beta) {
    const size_t flops = 2ull * M * N * K;
    if (metal_worthwhile(flops) && metal::sgemm(A, B, C, M, N, K, transA, transB, alpha, beta)) {
        return;
    }

    const blas_int m = narrow<blas_int>(M);
    const blas_int n = narrow<blas_int>(N);
    const blas_int k = narrow<blas_int>(K);
    const blas_int lda = transA ? m : k;
    const blas_int ldb = transB ? k : n;
    const blas_int ldc = n;

    cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void blas_sgemm(const float* A, const float* B, float* C, size_t M, size_t N, size_t K,
                       bool transA = false, bool transB = false) {
    blas_sgemm_ex(A, B, C, M, N, K, transA, transB, 1.0f, 0.0f);
}

// The CPU sgemm over strided row-major views: lda, ldb and ldc are the
// row pitches of A, B and C, so each can be a column slice of a wider
// matrix (one attention head of a (rows, d_model) projection). Never sent
// to the GPU, whose path needs whole page-aligned buffers.
inline void blas_sgemm_strided(bool transA, bool transB, size_t M, size_t N, size_t K, float alpha,
                               const float* A, size_t lda, const float* B, size_t ldb, float beta,
                               float* C, size_t ldc) {
    cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans, narrow<blas_int>(M), narrow<blas_int>(N),
                narrow<blas_int>(K), alpha, A, narrow<blas_int>(lda), B, narrow<blas_int>(ldb),
                beta, C, narrow<blas_int>(ldc));
}

// Element-wise transcendentals. Scalar libm calls dominate profiles for
// GELU and softmax; Accelerate's vForce computes them SIMD-wide.
// In-place (x == y) is allowed.
inline void vec_tanh(const float* x, float* y, size_t n) {
#if defined(__APPLE__)
    const int count = static_cast<int>(vec_length(n));
    vvtanhf(y, x, &count);
#else
    for (size_t i = 0; i < n; ++i) y[i] = std::tanh(x[i]);
#endif
}

inline void vec_exp(const float* x, float* y, size_t n) {
#if defined(__APPLE__)
    const int count = static_cast<int>(vec_length(n));
    vvexpf(y, x, &count);
#else
    for (size_t i = 0; i < n; ++i) y[i] = std::exp(x[i]);
#endif
}

// Sum of squares over a buffer (SIMD dot product with itself).
inline float vec_sum_squares(const float* x, size_t n) {
    return cblas_sdot(vec_length(n), x, 1, x, 1);
}

// Dot product of two contiguous buffers.
inline float vec_dot(const float* x, const float* y, size_t n) {
    return cblas_sdot(vec_length(n), x, 1, y, 1);
}

// y += alpha * x
inline void vec_axpy(float alpha, const float* x, float* y, size_t n) {
    cblas_saxpy(vec_length(n), alpha, x, 1, y, 1);
}

inline float vec_sum(const float* x, size_t n) {
#if defined(__APPLE__)
    float s;
    vDSP_sve(x, 1, &s, static_cast<vDSP_Length>(n));
    return s;
#else
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i) s += x[i];
    return s;
#endif
}

// x *= alpha
inline void vec_scale_inplace(float* x, float alpha, size_t n) {
    cblas_sscal(vec_length(n), alpha, x, 1);
}

// vDSP-style Vector Operations
inline void blas_vadd(const float* A, const float* B, float* C, size_t n) {
#if defined(__APPLE__)
    vDSP_vadd(A, 1, B, 1, C, 1, n);
#else
    for (size_t i = 0; i < n; ++i) C[i] = A[i] + B[i];
#endif
}

inline void blas_vsub(const float* A, const float* B, float* C, size_t n) {
#if defined(__APPLE__)
    vDSP_vsub(B, 1, A, 1, C, 1, n);
#else
    for (size_t i = 0; i < n; ++i) C[i] = A[i] - B[i];
#endif
}

inline void blas_vmul(const float* A, const float* B, float* C, size_t n) {
#if defined(__APPLE__)
    vDSP_vmul(A, 1, B, 1, C, 1, n);
#else
    for (size_t i = 0; i < n; ++i) C[i] = A[i] * B[i];
#endif
}

inline void blas_vsmul(const float* A, float scalar, float* C, size_t n) {
#if defined(__APPLE__)
    vDSP_vsmul(A, 1, &scalar, C, 1, n);
#else
    for (size_t i = 0; i < n; ++i) C[i] = A[i] * scalar;
#endif
}

inline void blas_vfill(float value, float* C, size_t n) {
#if defined(__APPLE__)
    vDSP_vfill(&value, C, 1, static_cast<vDSP_Length>(n));
#else
    for (size_t i = 0; i < n; ++i) C[i] = value;
#endif
}

}  // namespace grad
