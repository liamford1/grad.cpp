// Compares the Metal (MPS) sgemm against the CPU BLAS reference for the
// matrix shapes training actually uses, across all transpose combinations
// and the beta=1 accumulate contract. Skips (passes) when no Metal device
// is available - e.g. Linux, or macOS CI virtual machines.
#include "grad/transformer/metal_backend.h"
#include "grad/transformer/tensor.h"

#if defined(__APPLE__)
    #include <Accelerate/Accelerate.h>
#else
    #include <cblas.h>
#endif

#include <cmath>
#include <cstdio>
#include <random>

namespace {

void fill_random(Tensor& t, std::mt19937& gen) {
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    float* p = t.raw();
    for (size_t i = 0; i < t.numel(); i++) {
        p[i] = dis(gen);
    }
}

void cpu_reference(const float* A, const float* B, float* C,
                   int M, int N, int K, bool tA, bool tB,
                   float alpha, float beta) {
    cblas_sgemm(CblasRowMajor,
                tA ? CblasTrans : CblasNoTrans,
                tB ? CblasTrans : CblasNoTrans,
                M, N, K, alpha,
                A, tA ? M : K,
                B, tB ? K : N,
                beta, C, N);
}

bool run_case(int M, int N, int K, bool tA, bool tB,
              float alpha, float beta, std::mt19937& gen) {
    Tensor A(tA ? K : M, tA ? M : K);
    Tensor B(tB ? N : K, tB ? K : N);
    Tensor C_init(M, N);
    fill_random(A, gen);
    fill_random(B, gen);
    fill_random(C_init, gen);

    Tensor C_cpu = C_init;
    Tensor C_gpu = C_init;

    cpu_reference(A.raw(), B.raw(), C_cpu.raw(), M, N, K, tA, tB, alpha, beta);
    if (!metalgpu::sgemm(A.raw(), B.raw(), C_gpu.raw(), M, N, K, tA, tB, alpha, beta)) {
        std::printf("  FAIL M=%d N=%d K=%d tA=%d tB=%d: metal sgemm refused the call\n",
                    M, N, K, tA, tB);
        return false;
    }

    // Two valid fp32 GEMMs accumulate in different orders, so they agree
    // only to within accumulated rounding. That noise is ABSOLUTE (about
    // eps * K in the worst case for O(1) inputs), so the criterion must be
    // mixed: outputs near zero (catastrophic cancellation of K random
    // terms) are judged by absolute error, larger ones by relative error.
    // A pure relative criterion divides rounding noise by ~0 and fails
    // spuriously - verified against a float64 ground truth, which sits the
    // same distance from Accelerate as from MPS.
    //
    // With fp16 operands the dominant error is input rounding (~2^-11
    // relative per element, accumulating as ~sqrt(K) absolute for O(1)
    // inputs); accumulation itself stays fp32. Garbage-level bugs are
    // still orders of magnitude outside these bounds.
    const bool half_inputs = metalgpu::fp16_active();
    const float atol = half_inputs ? 5e-4f * std::sqrt(static_cast<float>(K)) + 1e-3f
                                   : 1e-6f * K + 1e-4f;
    const float rtol = half_inputs ? 1e-2f : 2e-3f;

    float worst = 0.0f;
    const float* cpu = C_cpu.raw();
    const float* gpu = C_gpu.raw();
    for (size_t i = 0; i < C_cpu.numel(); i++) {
        float tol = atol + rtol * (std::abs(cpu[i]) + std::abs(gpu[i]));
        worst = std::max(worst, std::abs(cpu[i] - gpu[i]) / tol);
    }

    bool ok = worst < 1.0f;
    std::printf("  %s M=%d N=%d K=%d tA=%d tB=%d alpha=%.1f beta=%.1f worst=%.3f of tolerance\n",
                ok ? "PASS" : "FAIL", M, N, K, tA, tB, alpha, beta, worst);
    return ok;
}

}  // namespace

int main() {
    std::printf("=== METAL MATMUL VS CPU BLAS ===\n");

    if (!metalgpu::available()) {
        std::printf("No Metal device available - skipping.\n");
        return 77;
    }
    std::printf("Operand precision: %s\n", metalgpu::fp16_active() ? "fp16" : "fp32");

    std::mt19937 gen(1234);
    int passed = 0, total = 0;

    struct Case { int M, N, K; bool tA, tB; float alpha, beta; };
    const Case cases[] = {
        // logits projection and its two backward forms
        {768, 5000, 512, false, true, 1.0f, 0.0f},
        {768, 512, 5000, false, false, 1.0f, 1.0f},
        {5000, 512, 768, true, false, 1.0f, 1.0f},
        // FFN forward/backward shapes
        {768, 2048, 512, false, false, 1.0f, 1.0f},
        {512, 2048, 768, true, false, 1.0f, 1.0f},
        {768, 512, 2048, false, true, 1.0f, 1.0f},
        // square projection shapes
        {768, 512, 512, false, false, 1.0f, 0.0f},
        {512, 512, 768, true, false, 1.0f, 1.0f},
    };

    for (const auto& c : cases) {
        total++;
        if (run_case(c.M, c.N, c.K, c.tA, c.tB, c.alpha, c.beta, gen)) passed++;
    }

    std::printf("Passed %d/%d\n", passed, total);
    return (passed == total) ? 0 : 1;
}
