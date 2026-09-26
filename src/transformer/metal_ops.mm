#include "grad/transformer/metal_backend.h"
#include "grad/transformer/metal_ops.h"
#include "grad/utils/narrow.h"
#include "metal_internal.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace grad::metal::ops {
namespace {

// Kernel parameter structs, mirroring metal_kernels.metal field for field.
struct EwParams {
    uint32_t n;
    float s;
};

struct GemmParams {
    uint32_t M, N, K;
    uint32_t lda, ldb, ldc;
    uint32_t transA, transB;
    float alpha, beta;
    uint32_t inner;
    uint32_t a_outer, a_inner, b_outer, b_inner, c_outer, c_inner;
};

constexpr NSUInteger kThreads1D = 256;

uint32_t u32(size_t n) {
    return narrow<uint32_t>(n);
}

BufferRef resolve(const void* p) {
    BufferRef ref;
    if (!pool().resolve(p, ref)) {
        throw std::logic_error(
            "Metal op on memory the Metal allocator does not own: tensors used in Metal mode "
            "must be allocated after set_device(Device::Metal)");
    }
    return ref;
}

// Collects a kernel's arguments, resolving every pointer before anything is
// encoded (a bad pointer throws with the stream untouched), then encodes the
// dispatch. Buffers take indices 0..n-1 in call order and the parameter
// struct index n.
class Launch {
public:
    explicit Launch(Kernel k) : kernel_(k) {}

    Launch& buf(const void* p) {
        if (count_ == bufs_.size()) throw std::logic_error("Metal launch: too many buffers");
        bufs_[count_++] = resolve(p);
        return *this;
    }

    template <typename P>
    Launch& params(const P& p) {
        static_assert(sizeof(P) <= sizeof(params_));
        std::memcpy(params_.data(), &p, sizeof(P));
        params_size_ = sizeof(P);
        return *this;
    }

    // One thread per element of an n-element grid.
    void threads(size_t n) { threads3(MTLSizeMake(n, 1, 1), kThreads1D); }
    // One thread per (x, y).
    void threads2(size_t x, size_t y) { threads3(MTLSizeMake(x, y, 1), kThreads1D); }
    // groups threadgroups of the kernel's required width.
    void groups(MTLSize groups) {
        encode([&](id<MTLComputeCommandEncoder> enc) {
            [enc dispatchThreadgroups:groups
                threadsPerThreadgroup:MTLSizeMake(kKernels[index()].threads, 1, 1)];
        });
    }

private:
    size_t index() const { return static_cast<size_t>(kernel_); }

    void threads3(MTLSize grid, NSUInteger width) {
        if (grid.width == 0 || grid.height == 0) return;
        encode([&](id<MTLComputeCommandEncoder> enc) {
            [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        });
    }

    template <typename Dispatch>
    void encode(Dispatch&& dispatch) {
        id<MTLComputePipelineState> pso = ctx().pipelines[index()];
        stream().compute([&](id<MTLComputeCommandEncoder> enc) {
            [enc setComputePipelineState:pso];
            for (size_t i = 0; i < count_; i++) {
                [enc setBuffer:bufs_[i].buffer offset:bufs_[i].offset atIndex:i];
            }
            if (params_size_ > 0) {
                [enc setBytes:params_.data() length:params_size_ atIndex:count_];
            }
            dispatch(enc);
        });
    }

    Kernel kernel_;
    std::array<BufferRef, 8> bufs_{};
    size_t count_ = 0;
    std::array<unsigned char, 128> params_{};
    size_t params_size_ = 0;
};

// MPS GEMM kernels are immutable objects bound to one shape and alpha/beta;
// creating one per call costs an allocation and an init, so they are cached
// (a model has a few dozen distinct GEMM shapes).
struct GemmKey {
    size_t M, N, K;
    bool transA, transB;
    float alpha, beta;
    bool operator==(const GemmKey&) const = default;
};

struct GemmKeyHash {
    size_t operator()(const GemmKey& k) const noexcept {
        size_t h = std::hash<size_t>{}(k.M);
        const auto mix = [&h](size_t v) { h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2); };
        mix(k.N);
        mix(k.K);
        mix((k.transA ? 1u : 0u) | (k.transB ? 2u : 0u));
        uint32_t a = 0;
        uint32_t b = 0;
        std::memcpy(&a, &k.alpha, sizeof(a));
        std::memcpy(&b, &k.beta, sizeof(b));
        mix((static_cast<size_t>(a) << 32) | b);
        return h;
    }
};

MPSMatrixMultiplication* mps_gemm(const GemmKey& key) {
    static std::mutex mu;
    static std::unordered_map<GemmKey, MPSMatrixMultiplication*, GemmKeyHash> cache;
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc] initWithDevice:ctx().device
                                                                    transposeLeft:key.transA
                                                                   transposeRight:key.transB
                                                                       resultRows:key.M
                                                                    resultColumns:key.N
                                                                  interiorColumns:key.K
                                                                            alpha:key.alpha
                                                                             beta:key.beta];
    if (!mm) throw std::runtime_error("cannot create an MPS matrix multiplication");
    cache.emplace(key, mm);
    return mm;
}

MPSMatrix* matrix(const BufferRef& ref, size_t rows, size_t cols) {
    MPSMatrixDescriptor* desc = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                                      columns:cols
                                                                     rowBytes:cols * sizeof(float)
                                                                     dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:ref.buffer offset:ref.offset descriptor:desc];
}

}  // namespace

void fill(float* y, size_t n, float value) {
    Launch(Kernel::Fill).buf(y).params(EwParams{u32(n), value}).threads(n);
}

void copy(const float* x, float* y, size_t n) {
    Launch(Kernel::Copy).buf(x).buf(y).params(EwParams{u32(n), 0.0f}).threads(n);
}

void gemm(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, bool transA,
          bool transB, float alpha, float beta) {
    if (M == 0 || N == 0) return;
    const BufferRef a = resolve(A);
    const BufferRef b = resolve(B);
    const BufferRef c = resolve(C);
    MPSMatrixMultiplication* mm = mps_gemm({M, N, K, transA, transB, alpha, beta});
    stream().command([&](id<MTLCommandBuffer> cb) {
        [mm encodeToCommandBuffer:cb
                       leftMatrix:matrix(a, transA ? K : M, transA ? M : K)
                      rightMatrix:matrix(b, transB ? N : K, transB ? K : N)
                     resultMatrix:matrix(c, M, N)];
    });
}

void gemm_batched(const BatchedGemm& g) {
    if (g.M == 0 || g.N == 0 || g.batch == 0) return;
    if (g.inner == 0) throw std::invalid_argument("gemm_batched: inner must be positive");
    const GemmParams p{u32(g.M),       u32(g.N),       u32(g.K),       u32(g.lda),
                       u32(g.ldb),     u32(g.ldc),     g.transA,       g.transB,
                       g.alpha,        g.beta,         u32(g.inner),   u32(g.a_outer),
                       u32(g.a_inner), u32(g.b_outer), u32(g.b_inner), u32(g.c_outer),
                       u32(g.c_inner)};
    constexpr size_t kTile = 32;
    Launch(Kernel::GemmBatched)
        .buf(g.A)
        .buf(g.B)
        .buf(g.C)
        .params(p)
        .groups(MTLSizeMake((g.N + kTile - 1) / kTile, (g.M + kTile - 1) / kTile, g.batch));
}

}  // namespace grad::metal::ops
