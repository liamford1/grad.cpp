#pragma once

// CBLAS enums (define BEFORE the extern "C")
enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};

extern "C" {
    // CBLAS interface for matrix multiplication
    void cblas_sgemm(const CBLAS_ORDER Order,
                     const CBLAS_TRANSPOSE TransA,
                     const CBLAS_TRANSPOSE TransB,
                     const int M, const int N, const int K,
                     const float alpha,
                     const float *A, const int lda,
                     const float *B, const int ldb,
                     const float beta,
                     float *C, const int ldc);
}