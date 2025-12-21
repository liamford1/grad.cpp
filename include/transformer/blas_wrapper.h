#pragma once

// Cross-platform BLAS + vDSP Wrapper
#if defined(__APPLE__)
    #include <Accelerate/Accelerate.h>
#endif

inline void blas_sgemm(const float* A, const float* B, float* C,
                       int M, int N, int K,
                       bool transA = false, bool transB = false)
{
#if defined(__APPLE__)
    int lda = transA ? M : K;
    int ldb = transB ? K : N;
    int ldc = N;

    cblas_sgemm(CblasRowMajor,
                transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans,
                M, N, K,
                1.0f,
                A, lda,
                B, ldb,
                0.0f,
                C, ldc);
#else
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[i*K + k] * B[k*N + j];
            C[i*N + j] = sum;
        }
    }
#endif
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

