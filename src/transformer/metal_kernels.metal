// Kernels of the resident stream (docs/design/metal-resident.md), compiled
// at runtime from this source, embedded into the library by CMake. ASCII
// only: CMake embeds it byte for byte.
//
// Every parameter struct mirrors one in metal_ops.mm field for field (32-bit
// fields only, so the layouts agree without packing rules). Extents fit in
// 32 bits: a Tensor holds at most 2^30 elements.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------- elementwise

struct EwParams {
    uint n;
    float s;
};

kernel void ew_fill(device float* y [[buffer(0)]], constant EwParams& p [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < p.n) y[i] = p.s;
}

kernel void ew_copy(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                    constant EwParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) y[i] = x[i];
}

// ---------------------------------------------------------------- batched GEMM

// C[z] = alpha * op(A[z]) @ op(B[z]) + beta * C[z] for z in [0, batch), where
// matrix z of each operand starts at (z / inner) * outer + (z % inner) *
// inner_stride. A threadgroup of four simdgroups computes a 32x32 tile of C,
// each simdgroup a 16x16 quarter as 2x2 8x8 simdgroup matrices, stepping K
// 16 at a time through threadgroup memory. The K loop order is fixed, so
// results do not depend on scheduling.
struct GemmParams {
    uint M, N, K;
    uint lda, ldb, ldc;
    uint transA, transB;
    float alpha, beta;
    uint inner;
    uint a_outer, a_inner, b_outer, b_inner, c_outer, c_inner;
};

constant constexpr uint kTileM = 32;
constant constexpr uint kTileN = 32;
constant constexpr uint kTileK = 16;
constant constexpr uint kGemmThreads = 128;

kernel void gemm_batched(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                         device float* C [[buffer(2)]], constant GemmParams& p [[buffer(3)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float As[kTileM * kTileK];
    threadgroup float Bs[kTileK * kTileN];
    threadgroup float Cs[kTileM * kTileN];

    const uint zo = group.z / p.inner;
    const uint zi = group.z % p.inner;
    A += zo * p.a_outer + zi * p.a_inner;
    B += zo * p.b_outer + zi * p.b_inner;
    C += zo * p.c_outer + zi * p.c_inner;

    const uint m0 = group.y * kTileM;
    const uint n0 = group.x * kTileN;
    const uint sm = (sg / 2) * 16;
    const uint sn = (sg % 2) * 16;

    simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < p.K; k0 += kTileK) {
        for (uint e = tid; e < kTileM * kTileK; e += kGemmThreads) {
            const uint i = e / kTileK;
            const uint k = e % kTileK;
            const uint gm = m0 + i;
            const uint gk = k0 + k;
            float v = 0.0f;
            if (gm < p.M && gk < p.K) v = p.transA ? A[gk * p.lda + gm] : A[gm * p.lda + gk];
            As[e] = v;
        }
        for (uint e = tid; e < kTileK * kTileN; e += kGemmThreads) {
            const uint k = e / kTileN;
            const uint j = e % kTileN;
            const uint gk = k0 + k;
            const uint gn = n0 + j;
            float v = 0.0f;
            if (gk < p.K && gn < p.N) v = p.transB ? B[gn * p.ldb + gk] : B[gk * p.ldb + gn];
            Bs[e] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < kTileK; kk += 8) {
            simdgroup_float8x8 a0, a1, b0, b1;
            simdgroup_load(a0, As + (sm + 0) * kTileK + kk, kTileK);
            simdgroup_load(a1, As + (sm + 8) * kTileK + kk, kTileK);
            simdgroup_load(b0, Bs + kk * kTileN + sn + 0, kTileN);
            simdgroup_load(b1, Bs + kk * kTileN + sn + 8, kTileN);
            simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
            simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
            simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
            simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc00, Cs + (sm + 0) * kTileN + sn + 0, kTileN);
    simdgroup_store(acc01, Cs + (sm + 0) * kTileN + sn + 8, kTileN);
    simdgroup_store(acc10, Cs + (sm + 8) * kTileN + sn + 0, kTileN);
    simdgroup_store(acc11, Cs + (sm + 8) * kTileN + sn + 8, kTileN);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = tid; e < kTileM * kTileN; e += kGemmThreads) {
        const uint gm = m0 + e / kTileN;
        const uint gn = n0 + e % kTileN;
        if (gm < p.M && gn < p.N) {
            float r = p.alpha * Cs[e];
            // beta = 0 must not read C: it may be uninitialized, and 0 * NaN
            // is NaN.
            if (p.beta != 0.0f) r += p.beta * C[gm * p.ldc + gn];
            C[gm * p.ldc + gn] = r;
        }
    }
}
