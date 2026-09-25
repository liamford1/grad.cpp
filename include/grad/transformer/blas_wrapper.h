#pragma once

// Cross-platform BLAS + vDSP Wrapper
#if defined(__APPLE__)
    #include <Accelerate/Accelerate.h>
#else
    #include <cblas.h>
#endif
#include "grad/transformer/metal_backend.h"
#include <cmath>
#include <cstdlib>

// Matmuls above this many FLOPs (2*M*N*K) route to the Metal GPU when one
// is available. The default crossover comes from measuring MPS against
// Accelerate on an M2 Pro across model scales (see BENCHMARKS.md): below
// ~10 GFLOPs the AMX units win or tie (the entire 22M-param config stays
// on CPU - measured, not assumed); above it the GPU pulls ahead, reaching
// 1.5-1.8x at 60-120M-param shapes. Tune with TRANSFORMER_METAL_THRESHOLD;
// disable with TRANSFORMER_METAL=0.
inline bool metal_worthwhile(size_t flops)
{
    static const long long threshold = [] {
        if (const char* env = std::getenv("TRANSFORMER_METAL_THRESHOLD")) {
            return static_cast<long long>(std::atoll(env));
        }
        return 10LL * 1000 * 1000 * 1000;
    }();
    return flops >= static_cast<size_t>(threshold);
}

// C = alpha * op(A) @ op(B) + beta * C
// A, B, C are row-major. op(A) is (M,K), op(B) is (K,N), C is (M,N).
// beta = 1 accumulates into C in place - used by backward passes to add
// gradient contributions without materializing a temporary.
//
// Large products try the GPU first. metalgpu::sgemm returns false only
// when it submitted nothing and left C untouched, which is what makes
// rerunning on the CPU with the same beta correct; a GPU failure after
// submission throws instead of coming back here.
inline void blas_sgemm_ex(const float* A, const float* B, float* C,
                          int M, int N, int K,
                          bool transA, bool transB,
                          float alpha, float beta)
{
    const size_t flops = 2ull * M * N * K;
    if (metal_worthwhile(flops)
        && metalgpu::sgemm(A, B, C, M, N, K, transA, transB, alpha, beta)) {
        return;
    }

    int lda = transA ? M : K;
    int ldb = transB ? K : N;
    int ldc = N;

    cblas_sgemm(CblasRowMajor,
                transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans,
                M, N, K,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc);
}

inline void blas_sgemm(const float* A, const float* B, float* C,
                       int M, int N, int K,
                       bool transA = false, bool transB = false)
{
    blas_sgemm_ex(A, B, C, M, N, K, transA, transB, 1.0f, 0.0f);
}

// Element-wise transcendentals. Scalar libm calls dominate profiles for
// GELU and softmax; Accelerate's vForce computes them SIMD-wide.
// In-place (x == y) is allowed.
inline void vec_tanh(const float* x, float* y, int n)
{
#if defined(__APPLE__)
    vvtanhf(y, x, &n);
#else
    for (int i = 0; i < n; ++i) y[i] = std::tanh(x[i]);
#endif
}

inline void vec_exp(const float* x, float* y, int n)
{
#if defined(__APPLE__)
    vvexpf(y, x, &n);
#else
    for (int i = 0; i < n; ++i) y[i] = std::exp(x[i]);
#endif
}

// Sum of squares over a buffer (SIMD dot product with itself).
inline float vec_sum_squares(const float* x, int n)
{
    return cblas_sdot(n, x, 1, x, 1);
}

inline float vec_sum(const float* x, int n)
{
#if defined(__APPLE__)
    float s;
    vDSP_sve(x, 1, &s, static_cast<vDSP_Length>(n));
    return s;
#else
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += x[i];
    return s;
#endif
}

// x *= alpha
inline void vec_scale_inplace(float* x, float alpha, int n)
{
    cblas_sscal(n, alpha, x, 1);
}

// vDSP-style Vector Operations
inline void blas_vadd(const float* A, const float* B, float* C, size_t n)
{
#if defined(__APPLE__)
    vDSP_vadd(A, 1, B, 1, C, 1, n);
#else
    for (size_t i = 0; i < n; ++i)
        C[i] = A[i] + B[i];
#endif
}

inline void blas_vsub(const float* A, const float* B, float* C, size_t n)
{
#if defined(__APPLE__)
    vDSP_vsub(B, 1, A, 1, C, 1, n);
#else
    for (size_t i = 0; i < n; ++i)
        C[i] = A[i] - B[i];
#endif
}

inline void blas_vmul(const float* A, const float* B, float* C, size_t n)
{
#if defined(__APPLE__)
    vDSP_vmul(A, 1, B, 1, C, 1, n);
#else
    for (size_t i = 0; i < n; ++i)
        C[i] = A[i] * B[i];
#endif
}

inline void blas_vsmul(const float* A, float scalar, float* C, size_t n)
{
#if defined(__APPLE__)
    vDSP_vsmul(A, 1, &scalar, C, 1, n);
#else
    for (size_t i = 0; i < n; ++i)
        C[i] = A[i] * scalar;
#endif
}

inline void blas_vfill(float value, float* C, size_t n)
{
#if defined(__APPLE__)
    vDSP_vfill(&value, C, 1, static_cast<vDSP_Length>(n));
#else
    for (size_t i = 0; i < n; ++i)
        C[i] = value;
#endif
}

