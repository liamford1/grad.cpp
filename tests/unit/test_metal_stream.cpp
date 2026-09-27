// The resident Metal stream (docs/design/metal-resident.md): GEMMs and
// kernels encoded without waiting, the CPU-access fence, deferred reuse of
// freed storage, and the batched strided GEMM attention uses. Exits 77
// (skipped) without a Metal device, e.g. on Linux or a macOS CI VM.
#include "grad/transformer/blas_wrapper.h"
#include "grad/transformer/device.h"
#include "grad/transformer/metal_backend.h"
#include "grad/transformer/metal_ops.h"
#include "grad/transformer/tensor.h"
#include "../test_util.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace grad;

namespace {

void fill_random(Tensor& t, std::mt19937& gen) {
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (float& v : t.values()) v = dis(gen);
}

// Two fp32 GEMMs that accumulate in different orders agree to within
// absolute rounding noise of about eps * K for O(1) inputs, plus a relative
// term for large outputs; see test_metal_matmul.cpp for the derivation.
float worst_error(const float* expected, const float* actual, size_t n, size_t K) {
    const float atol = 1e-6f * static_cast<float>(K) + 1e-4f;
    const float rtol = 2e-3f;
    float worst = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float tol = atol + rtol * (std::abs(expected[i]) + std::abs(actual[i]));
        worst = std::max(worst, std::abs(expected[i] - actual[i]) / tol);
    }
    return worst;
}

uint64_t syncs() {
    return metal::stream_stats().syncs;
}

// A chain of dependent GEMMs is encoded with no wait, and the first CPU
// read of the result waits exactly once and sees the final values.
void test_gemm_chain_without_waits(std::mt19937& gen) {
    std::cout << "gemm chain without waits" << std::endl;
    constexpr size_t n = 96;
    Tensor a(n, n);
    Tensor b(n, n);
    fill_random(a, gen);
    fill_random(b, gen);
    // The CPU reference first: once b has been handed to the GPU, reading
    // it would itself wait.
    constexpr int kSteps = 8;
    std::vector<float> ref(a.values().begin(), a.values().end());
    std::vector<float> tmp(n * n);
    for (int step = 0; step < kSteps; step++) {
        blas_sgemm_strided(false, false, n, n, n, 0.5f, ref.data(), n, b.raw(), n, 0.0f, tmp.data(),
                           n);
        ref.swap(tmp);
    }

    const uint64_t before = syncs();
    Tensor x = a;  // a is not GPU-visible yet, so this is a CPU copy
    Tensor y = Tensor::uninitialized(n, n);
    for (int step = 0; step < kSteps; step++) {
        metal::ops::gemm(x.device_data(), b.device_data(), y.device_data(), n, n, n, false, false,
                         0.5f, 0.0f);
        swap(x, y);
    }
    CHECK(syncs() == before);
    const float worst = worst_error(ref.data(), x.raw(), n * n, n * kSteps);
    CHECK(syncs() >= before + 1);
    std::cout << "  worst " << worst << " of tolerance" << std::endl;
    CHECK(worst < 1.0f);
}

// A CPU write to a tensor with a queued GPU writer waits for it first.
void test_cpu_write_waits_for_gpu() {
    std::cout << "cpu write waits for queued gpu writes" << std::endl;
    constexpr size_t n = size_t{1} << 20;
    Tensor t = Tensor::uninitialized(Shape{n});
    metal::ops::fill(t.device_data(), n, 1.0f);
    t.raw()[0] = 5.0f;  // must land after the fill, not be overwritten by it
    metal::synchronize();
    CHECK(t.raw()[0] == 5.0f);
    CHECK(t.raw()[n - 1] == 1.0f);
}

// A tensor never handed to the GPU is read and written without waiting,
// even while GPU work is queued.
void test_unshared_tensors_do_not_wait() {
    std::cout << "tensors not given to the gpu never wait" << std::endl;
    constexpr size_t n = 256;
    Tensor gpu_side(n, n);
    metal::ops::fill(gpu_side.device_data(), n * n, 2.0f);
    const uint64_t before = syncs();
    Tensor host = Tensor::uninitialized(n, n);
    for (float& v : host.values()) v = 3.0f;
    CHECK(host.getValue(1, 2) == 3.0f);
    CHECK(syncs() == before);
    CHECK(gpu_side.getValue(1, 2) == 2.0f);  // this one waits
    CHECK(syncs() == before + 1);
}

// Storage freed while a queued kernel still reads it is not handed out
// again until the GPU is done, so scribbling on a new tensor cannot corrupt
// the queued computation.
void test_deferred_reuse(std::mt19937& gen) {
    std::cout << "freed storage is not reused while queued work reads it" << std::endl;
    constexpr size_t n = 128;
    Tensor b(n, n);
    fill_random(b, gen);
    Tensor c = Tensor::uninitialized(n, n);
    std::vector<float> ref(n * n);
    const float* old_ptr = nullptr;
    {
        Tensor a(n, n);
        fill_random(a, gen);
        blas_sgemm_strided(false, false, n, n, n, 1.0f, a.raw(), n, b.raw(), n, 0.0f, ref.data(),
                           n);
        old_ptr = a.device_data();
        metal::ops::gemm(a.device_data(), b.device_data(), c.device_data(), n, n, n, false, false,
                         1.0f, 0.0f);
    }  // a released with the GEMM still queued
    Tensor scribble = Tensor::uninitialized(n, n);
    CHECK(scribble.raw() != old_ptr);
    for (float& v : scribble.values()) v = 1e30f;
    CHECK(worst_error(ref.data(), c.raw(), n * n, n) < 1.0f);
    // Once the GPU is idle the block is reusable.
    Tensor again = Tensor::uninitialized(n, n);
    (void)again;
}

// The batched strided GEMM against CPU BLAS on attention-shaped problems:
// heads as column slices of (B*S, d) matrices, every transpose, beta = 1,
// and sizes that are not tile multiples.
void test_batched_gemm(std::mt19937& gen) {
    std::cout << "batched strided gemm" << std::endl;
    struct Case {
        size_t B, H, S, hs;
        bool tA, tB;
        float beta;
    };
    const Case cases[] = {
        {2, 3, 37, 16, false, true, 0.0f},   // scores = Q K^T
        {2, 3, 37, 16, false, false, 0.0f},  // attended = P V
        {2, 3, 37, 16, true, false, 1.0f},   // dV = P^T dA, accumulated
        {1, 4, 96, 64, false, true, 0.0f},  {3, 2, 5, 8, true, false, 0.0f},
    };
    for (const Case& c : cases) {
        const size_t d = c.H * c.hs;
        const size_t units = c.B * c.H;
        // Q-like operand (B*S, d) and a per-unit (S, S) operand.
        Tensor proj(c.B * c.S, d);
        Tensor sq(units, c.S, c.S);
        fill_random(proj, gen);
        fill_random(sq, gen);
        const bool scores = c.tB;  // (S, hs) x (S, hs)^T -> (S, S)
        Tensor out = scores ? Tensor(units, c.S, c.S) : Tensor(c.B * c.S, d);
        fill_random(out, gen);
        std::vector<float> ref(out.values().begin(), out.values().end());

        metal::ops::BatchedGemm g;
        g.batch = units;
        g.inner = c.H;
        g.transA = c.tA;
        g.transB = c.tB;
        g.beta = c.beta;
        if (scores) {
            g.M = c.S, g.N = c.S, g.K = c.hs;
            g.A = proj.device_data(), g.lda = d, g.a_outer = c.S * d, g.a_inner = c.hs;
            g.B = proj.device_data(), g.ldb = d, g.b_outer = c.S * d, g.b_inner = c.hs;
            g.C = out.device_data(), g.ldc = c.S, g.c_outer = c.H * c.S * c.S,
            g.c_inner = c.S * c.S;
        } else {
            g.M = c.S, g.N = c.hs, g.K = c.S;
            g.A = sq.device_data(), g.lda = c.S, g.a_outer = c.H * c.S * c.S, g.a_inner = c.S * c.S;
            g.B = proj.device_data(), g.ldb = d, g.b_outer = c.S * d, g.b_inner = c.hs;
            g.C = out.device_data(), g.ldc = d, g.c_outer = c.S * d, g.c_inner = c.hs;
        }
        metal::ops::gemm_batched(g);

        const float* P = proj.raw();
        const float* Sq = sq.raw();
        for (size_t u = 0; u < units; u++) {
            const size_t b = u / c.H;
            const size_t h = u % c.H;
            const float* a_ptr = scores ? P + b * c.S * d + h * c.hs : Sq + u * c.S * c.S;
            const float* b_ptr = P + b * c.S * d + h * c.hs;
            float* c_ptr =
                scores ? ref.data() + u * c.S * c.S : ref.data() + b * c.S * d + h * c.hs;
            blas_sgemm_strided(c.tA, c.tB, g.M, g.N, g.K, 1.0f, a_ptr, g.lda, b_ptr, g.ldb, c.beta,
                               c_ptr, g.ldc);
        }
        const float worst = worst_error(ref.data(), out.raw(), out.numel(), g.K);
        std::cout << "  B=" << c.B << " H=" << c.H << " S=" << c.S << " hs=" << c.hs
                  << " tA=" << c.tA << " tB=" << c.tB << " beta=" << c.beta << ": worst " << worst
                  << std::endl;
        CHECK(worst < 1.0f);
    }
}

// MPS GEMMs at shapes that are not multiples of anything (the tiny test
// models have vocab 97), every transpose, beta = 1.
void test_mps_odd_shapes(std::mt19937& gen) {
    std::cout << "mps gemm, odd shapes" << std::endl;
    struct Case {
        size_t M, N, K;
        bool tA, tB;
        float beta;
    };
    const Case cases[] = {{64, 97, 16, false, true, 0.0f},
                          {64, 16, 97, false, false, 1.0f},
                          {97, 16, 64, true, false, 1.0f},
                          {5, 3, 7, true, true, 0.0f}};
    for (const Case& c : cases) {
        Tensor A(c.tA ? c.K : c.M, c.tA ? c.M : c.K);
        Tensor B(c.tB ? c.N : c.K, c.tB ? c.K : c.N);
        Tensor C(c.M, c.N);
        fill_random(A, gen);
        fill_random(B, gen);
        fill_random(C, gen);
        std::vector<float> ref(C.values().begin(), C.values().end());
        blas_sgemm_strided(c.tA, c.tB, c.M, c.N, c.K, 1.0f, A.raw(), c.tA ? c.M : c.K, B.raw(),
                           c.tB ? c.K : c.N, c.beta, ref.data(), c.N);
        metal::ops::gemm(A.device_data(), B.device_data(), C.device_data(), c.M, c.N, c.K, c.tA,
                         c.tB, 1.0f, c.beta);
        const float worst = worst_error(ref.data(), C.raw(), C.numel(), c.K);
        std::cout << "  M=" << c.M << " N=" << c.N << " K=" << c.K << ": worst " << worst
                  << std::endl;
        CHECK(worst < 1.0f);
    }
}

// The same GEMMs twice give bitwise identical results.
void test_gemm_determinism(std::mt19937& gen) {
    std::cout << "gemm determinism" << std::endl;
    Tensor q(200, 48);
    Tensor w(48, 200);
    fill_random(q, gen);
    fill_random(w, gen);
    auto run = [&] {
        Tensor out = Tensor::uninitialized(200, 200);
        metal::ops::gemm(q.device_data(), w.device_data(), out.device_data(), 200, 200, 48, false,
                         false, 1.0f, 0.0f);
        Tensor scores = Tensor::uninitialized(12, 50, 50);
        metal::ops::BatchedGemm g;
        g.M = 50, g.N = 50, g.K = 16, g.transB = true, g.batch = 12, g.inner = 3;
        g.A = q.device_data(), g.lda = 48, g.a_outer = 2400, g.a_inner = 16;
        g.B = q.device_data(), g.ldb = 48, g.b_outer = 2400, g.b_inner = 16;
        g.C = scores.device_data(), g.ldc = 50, g.c_outer = 7500, g.c_inner = 2500;
        metal::ops::gemm_batched(g);
        std::vector<float> all(out.values().begin(), out.values().end());
        all.insert(all.end(), scores.values().begin(), scores.values().end());
        return all;
    };
    const auto first = run();
    const auto second = run();
    CHECK(std::memcmp(first.data(), second.data(), first.size() * sizeof(float)) == 0);
}

// Large zero-filled tensors are cleared by the GPU, copies of GPU-visible
// tensors are made by the GPU, and neither waits.
void test_constructors() {
    std::cout << "zero fill and copy in metal mode" << std::endl;
    constexpr size_t n = size_t{1} << 18;  // above the CPU zero-fill cutoff
    Tensor seed = Tensor::uninitialized(Shape{n});
    metal::ops::fill(seed.device_data(), n, 7.0f);
    const uint64_t before = syncs();
    Tensor zeros(Shape{n});
    Tensor copy = seed;
    CHECK(syncs() == before);
    CHECK(zeros.device_visible());
    CHECK(copy.device_visible());
    CHECK(zeros.raw()[n / 2] == 0.0f);
    CHECK(copy.raw()[n - 1] == 7.0f);
}

void test_foreign_memory_rejected() {
    std::cout << "memory not from the metal allocator is rejected" << std::endl;
    std::vector<float> host(64);
    bool threw = false;
    try {
        metal::ops::fill(host.data(), host.size(), 1.0f);
    } catch (const std::logic_error&) {
        threw = true;
    }
    CHECK(threw);
}

}  // namespace

int main() {
    std::cout << "=== METAL RESIDENT STREAM ===" << std::endl;
    if (!metal::resident_available()) {
        std::cout << "Resident Metal unavailable (" << metal::resident_status() << "), skipping."
                  << std::endl;
        return 77;
    }
    std::mt19937 gen(4321);

    // CPU mode never touches the stream.
    {
        const auto before = metal::stream_stats();
        Tensor t(64, 64);
        t.fill(1.0f);
        CHECK(!t.device_visible());
        CHECK(metal::stream_stats().dispatches == before.dispatches);
    }

    set_device(Device::Metal);
    CHECK(metal_mode());
    try {
        test_gemm_chain_without_waits(gen);
        test_cpu_write_waits_for_gpu();
        test_unshared_tensors_do_not_wait();
        test_deferred_reuse(gen);
        test_batched_gemm(gen);
        test_mps_odd_shapes(gen);
        test_gemm_determinism(gen);
        test_constructors();
        test_foreign_memory_rejected();
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << std::endl;
        CHECK(false);
    }

    // Leaving Metal mode drains the stream.
    Tensor last = Tensor::uninitialized(Shape{1024});
    metal::ops::fill(last.device_data(), 1024, 4.0f);
    set_device(Device::CPU);
    const auto stats = metal::stream_stats();
    std::cout << "syncs " << stats.syncs << ", command buffers " << stats.command_buffers
              << ", dispatches " << stats.dispatches << ", peak live "
              << stats.peak_live_bytes / 1024 << " KB" << std::endl;
    CHECK(last.raw()[1023] == 4.0f);
    return test_util::exit_code();
}
