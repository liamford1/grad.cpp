#pragma once

// Cross-platform BLAS + vDSP Wrapper
#if defined(__APPLE__)
    #include <Accelerate/Accelerate.h>
#else
    #include <cblas.h>
#endif

// C = alpha * op(A) @ op(B) + beta * C
// A, B, C are row-major. op(A) is (M,K), op(B) is (K,N), C is (M,N).
// beta = 1 accumulates into C in place - used by backward passes to add
// gradient contributions without materializing a temporary.
inline void blas_sgemm_ex(const float* A, const float* B, float* C,
                          int M, int N, int K,
                          bool transA, bool transB,
                          float alpha, float beta)
{
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

