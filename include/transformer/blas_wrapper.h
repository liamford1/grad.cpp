#pragma once

enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};

extern "C" {
    void cblas_sgemm(const CBLAS_ORDER Order,
                     const CBLAS_TRANSPOSE TransA,
                     const CBLAS_TRANSPOSE TransB,
                     const int M, const int N, const int K,
                     const float alpha,
                     const float *A, const int lda,
                     const float *B, const int ldb,
                     const float beta,
                     float *C, const int ldc);
    
    void vDSP_vadd(const float *A, long strideA,
                   const float *B, long strideB,
                   float *C, long strideC,
                   unsigned long n);
    
    void vDSP_vsub(const float *B, long strideB,
                   const float *A, long strideA,
                   float *C, long strideC,
                   unsigned long n);
    
    void vDSP_vmul(const float *A, long strideA,
                   const float *B, long strideB,
                   float *C, long strideC,
                   unsigned long n);
    
    void vDSP_vsmul(const float *A, long strideA,
                    const float *B,
                    float *C, long strideC,
                    unsigned long n);
}