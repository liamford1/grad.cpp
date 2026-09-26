#include "grad/transformer/metal_backend.h"
#include "grad/transformer/metal_ops.h"
#include "grad/utils/narrow.h"
#include "metal_internal.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <span>
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

struct RowsParams {
    uint32_t n, cols, period;
};

struct BroadcastParams {
    uint32_t batch, rows, cols;
    uint32_t a_batch, a_row, a_col;
    uint32_t b_batch, b_row, b_col;
};

struct ReduceParams {
    uint32_t batch, rows, cols;
    uint32_t dst_rows, dst_cols;
    uint32_t br, bc;
};

struct ColParams {
    uint32_t rows, cols, block_rows, nblocks;
};

struct RowParams {
    uint32_t rows, cols;
    float eps;
    uint32_t flag;
    float scale;
};

struct NormParamsParams {
    uint32_t rows, d, block_rows, nblocks, rms, want_beta;
};

struct EmbedParams {
    uint32_t n, d;
    float scale;
};

struct RopeParams {
    uint32_t rows, S, d, half_head, inverse;
};

struct DropoutParams {
    uint64_t seed;
    uint64_t first_stream;
    uint32_t n, units, blocks, threshold;
    float scale;
    uint32_t has_x;
};

struct AdamParams {
    uint32_t n;
    float b1, b2, inv_bc1, inv_bc2, lr, eps, wd;
};

struct SumsqParams {
    uint32_t n, offset;
};

struct NormParams {
    uint32_t count;
    float max_norm;
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
// Rows per block in the two-pass column sums: (B*S, d) gradients of a few
// thousand rows split into ~64 blocks, so d * 64 threads share the work.
constexpr size_t kColBlockRows = 32;
constexpr size_t kNormChunk = 4096;  // elements per sumsq_partial threadgroup
constexpr size_t kDropBlock = 65536;

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

    Launch& buf(const void* p) { return raw_buf(resolve(p)); }

    // A buffer the backend owns outside the pool (the dropout jump table).
    Launch& raw_buf(const BufferRef& ref) {
        if (count_ == bufs_.size()) throw std::logic_error("Metal launch: too many buffers");
        bufs_[count_++] = ref;
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

// GPU scratch for one op: released at scope end, which the pool defers
// until the kernels just encoded have run.
class Scratch {
public:
    explicit Scratch(size_t n) : p_(pool().allocate(n * sizeof(float))) {}
    ~Scratch() { pool().release(p_); }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;
    [[nodiscard]] float* get() const { return p_; }

private:
    float* p_;
};

void elementwise(Kernel k, const float* a, float* y, size_t n, float s = 0.0f) {
    Launch(k).buf(a).buf(y).params(EwParams{u32(n), s}).threads(n);
}

void elementwise3(Kernel k, const float* a, const float* b, float* y, size_t n) {
    Launch(k).buf(a).buf(b).buf(y).params(EwParams{u32(n), 0.0f}).threads(n);
}

void rows_kernel(Kernel k, const float* x, float* y, size_t rows, size_t cols) {
    Launch(k)
        .buf(x)
        .buf(y)
        .params(RowParams{u32(rows), u32(cols), 0.0f, 0, 0.0f})
        .groups(MTLSizeMake(rows, 1, 1));
}

void rows_kernel3(Kernel k, const float* a, const float* b, float* y, size_t rows, size_t cols) {
    Launch(k)
        .buf(a)
        .buf(b)
        .buf(y)
        .params(RowParams{u32(rows), u32(cols), 0.0f, 0, 0.0f})
        .groups(MTLSizeMake(rows, 1, 1));
}

// dst[c] += sum over nblocks rows of part (nblocks, cols), in order.
void col_finish(const float* part, size_t nblocks, size_t cols, float* dst) {
    Launch(Kernel::ColFinish)
        .buf(part)
        .buf(dst)
        .params(ColParams{u32(nblocks), u32(cols), 1, u32(nblocks)})
        .threads(cols);
}

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
    elementwise(Kernel::Copy, x, y, n);
}

void add(const float* a, const float* b, float* y, size_t n) {
    elementwise3(Kernel::Add, a, b, y, n);
}

void accumulate(float* y, const float* x, size_t n) {
    elementwise(Kernel::Acc, x, y, n);
}

void accumulate_scaled(float* y, const float* x, float s, size_t n) {
    elementwise(Kernel::AccScaled, x, y, n, s);
}

void mul(const float* a, const float* b, float* y, size_t n) {
    elementwise3(Kernel::Mul, a, b, y, n);
}

void mul_accumulate(float* y, const float* a, const float* b, size_t n) {
    elementwise3(Kernel::MulAcc, a, b, y, n);
}

void scale(const float* x, float s, float* y, size_t n) {
    elementwise(Kernel::Scale, x, y, n, s);
}

void scale_by(float* y, const float* s, size_t n) {
    elementwise(Kernel::ScaleBy, s, y, n);
}

void add_rows(const float* x, const float* rows, float* y, size_t n, size_t cols, size_t period) {
    Launch(Kernel::AddRows)
        .buf(x)
        .buf(rows)
        .buf(y)
        .params(RowsParams{u32(n), u32(cols), u32(period)})
        .threads(n);
}

void broadcast_add(const float* a, Broadcast sa, const float* b, Broadcast sb, float* y,
                   size_t batch, size_t rows, size_t cols) {
    const BroadcastParams p{u32(batch),  u32(rows),     u32(cols),   u32(sa.batch), u32(sa.row),
                            u32(sa.col), u32(sb.batch), u32(sb.row), u32(sb.col)};
    Launch(Kernel::BroadcastAdd).buf(a).buf(b).buf(y).params(p).threads(batch * rows * cols);
}

void broadcast_reduce(const float* g, size_t batch, size_t rows, size_t cols, float* dst,
                      size_t dst_rows, size_t dst_cols, bool br, bool bc) {
    const ReduceParams p{u32(batch),    u32(rows),    u32(cols),   u32(dst_rows),
                         u32(dst_cols), br ? 1u : 0u, bc ? 1u : 0u};
    Launch(Kernel::BroadcastReduce).buf(g).buf(dst).params(p).threads(dst_rows * dst_cols);
}

void column_sums(const float* x, size_t rows, size_t cols, float* dst) {
    if (rows == 0 || cols == 0) return;
    // Few rows (a batch of position-table gradients): one pass suffices.
    if (rows <= 2 * kColBlockRows) {
        col_finish(x, rows, cols, dst);
        return;
    }
    const size_t nblocks = (rows + kColBlockRows - 1) / kColBlockRows;
    const Scratch part(nblocks * cols);
    Launch(Kernel::ColPartial)
        .buf(x)
        .buf(part.get())
        .params(ColParams{u32(rows), u32(cols), u32(kColBlockRows), u32(nblocks)})
        .threads2(cols, nblocks);
    col_finish(part.get(), nblocks, cols, dst);
}

void gelu(const float* x, float* y, size_t n) {
    elementwise(Kernel::GeluFwd, x, y, n);
}

void gelu_backward(const float* x, const float* dy, float* dx, size_t n) {
    elementwise3(Kernel::GeluBwd, x, dy, dx, n);
}

void silu(const float* x, float* y, size_t n) {
    elementwise(Kernel::SiluFwd, x, y, n);
}

void silu_backward(const float* x, const float* dy, float* dx, size_t n) {
    elementwise3(Kernel::SiluBwd, x, dy, dx, n);
}

void layer_norm(const float* x, const float* gamma, const float* beta, float* y, float* mean,
                float* rstd, size_t rows, size_t d, float eps, bool rms) {
    Launch(Kernel::LayerNormFwd)
        .buf(x)
        .buf(gamma)
        .buf(beta)
        .buf(y)
        .buf(mean)
        .buf(rstd)
        .params(RowParams{u32(rows), u32(d), eps, rms ? 1u : 0u, 0.0f})
        .groups(MTLSizeMake(rows, 1, 1));
}

void layer_norm_backward(const float* x, const float* gamma, const float* dy, const float* mean,
                         const float* rstd, float* dx, float* dgamma, float* dbeta, size_t rows,
                         size_t d, bool rms) {
    if (rows == 0) return;
    if (dx) {
        Launch(Kernel::LayerNormBwdDx)
            .buf(x)
            .buf(gamma)
            .buf(dy)
            .buf(mean)
            .buf(rstd)
            .buf(dx)
            .params(RowParams{u32(rows), u32(d), 0.0f, rms ? 1u : 0u, 0.0f})
            .groups(MTLSizeMake(rows, 1, 1));
    }
    const bool want_beta = dbeta != nullptr && !rms;
    if (!dgamma && !want_beta) return;
    const size_t nblocks = (rows + kColBlockRows - 1) / kColBlockRows;
    const Scratch gpart(nblocks * d);
    const Scratch bpart(want_beta ? nblocks * d : 1);
    Launch(Kernel::LayerNormBwdParams)
        .buf(x)
        .buf(dy)
        .buf(mean)
        .buf(rstd)
        .buf(gpart.get())
        .buf(bpart.get())
        .params(NormParamsParams{u32(rows), u32(d), u32(kColBlockRows), u32(nblocks), rms ? 1u : 0u,
                                 want_beta ? 1u : 0u})
        .threads2(d, nblocks);
    if (dgamma) col_finish(gpart.get(), nblocks, d, dgamma);
    if (want_beta) col_finish(bpart.get(), nblocks, d, dbeta);
}

void softmax(const float* x, float* y, size_t rows, size_t cols) {
    rows_kernel(Kernel::SoftmaxFwd, x, y, rows, cols);
}

void softmax_backward(const float* y, const float* dy, float* dx, size_t rows, size_t cols) {
    rows_kernel3(Kernel::SoftmaxBwd, y, dy, dx, rows, cols);
}

void log_softmax(const float* x, float* y, size_t rows, size_t cols) {
    rows_kernel(Kernel::LogSoftmaxFwd, x, y, rows, cols);
}

void log_softmax_backward(const float* y, const float* dy, float* dx, size_t rows, size_t cols) {
    rows_kernel3(Kernel::LogSoftmaxBwd, y, dy, dx, rows, cols);
}

void attention_softmax(float* scores, size_t rows, size_t S, float scale) {
    Launch(Kernel::AttnSoftmaxFwd)
        .buf(scores)
        .params(RowParams{u32(rows), u32(S), 0.0f, 0, scale})
        .groups(MTLSizeMake(rows, 1, 1));
}

void attention_softmax_backward(const float* P, float* dp, const float* mask, size_t rows, size_t S,
                                float scale) {
    Launch(Kernel::AttnSoftmaxBwd)
        .buf(P)
        .buf(dp)
        .buf(mask ? mask : P)
        .params(RowParams{u32(rows), u32(S), 0.0f, mask ? 1u : 0u, scale})
        .groups(MTLSizeMake(rows, 1, 1));
}

void nll_loss(const float* logp, const float* targets, float* loss, size_t rows, size_t vocab) {
    Launch(Kernel::NllFwd)
        .buf(logp)
        .buf(targets)
        .buf(loss)
        .params(RowParams{u32(rows), u32(vocab), 0.0f, 0, 0.0f})
        .groups(MTLSizeMake(1, 1, 1));
}

void nll_loss_backward(const float* g, const float* targets, float* dlogp, size_t rows,
                       size_t vocab) {
    Launch(Kernel::NllBwd)
        .buf(g)
        .buf(targets)
        .buf(dlogp)
        .params(RowParams{u32(rows), u32(vocab), 0.0f, 0, 0.0f})
        .threads(rows);
}

void cross_entropy(const float* logits, const float* targets, float* stats, float* row_loss,
                   float* loss, size_t rows, size_t vocab) {
    Launch(Kernel::CeFwd)
        .buf(logits)
        .buf(targets)
        .buf(stats)
        .buf(row_loss)
        .params(RowParams{u32(rows), u32(vocab), 0.0f, 0, 0.0f})
        .groups(MTLSizeMake(rows, 1, 1));
    Launch(Kernel::MeanReduce)
        .buf(row_loss)
        .buf(loss)
        .params(EwParams{u32(rows), 0.0f})
        .groups(MTLSizeMake(1, 1, 1));
}

void cross_entropy_backward(const float* logits, const float* stats, const float* targets,
                            const float* g, float* dlogits, size_t rows, size_t vocab) {
    Launch(Kernel::CeBwd)
        .buf(logits)
        .buf(stats)
        .buf(targets)
        .buf(g)
        .buf(dlogits)
        .params(RowParams{u32(rows), u32(vocab), 0.0f, 0, 0.0f})
        .threads(rows * vocab);
}

void embedding(const float* ids, const float* table, float* out, size_t tokens, size_t d,
               float scale) {
    Launch(Kernel::EmbeddingFwd)
        .buf(ids)
        .buf(table)
        .buf(out)
        .params(EmbedParams{u32(tokens * d), u32(d), scale})
        .threads(tokens * d);
}

void embedding_backward(const float* tokens, const float* starts, const float* order, size_t unique,
                        const float* dout, float* dtable, size_t d, float scale) {
    Launch(Kernel::EmbeddingBwd)
        .buf(tokens)
        .buf(starts)
        .buf(order)
        .buf(dout)
        .buf(dtable)
        .params(EmbedParams{u32(unique * d), u32(d), scale})
        .threads(unique * d);
}

void rope(float* x, size_t rows, size_t S, size_t d, size_t head, const float* cos_t,
          const float* sin_t, bool inverse) {
    Launch(Kernel::Rope)
        .buf(x)
        .buf(cos_t)
        .buf(sin_t)
        .params(RopeParams{u32(rows), u32(S), u32(d), u32(head / 2), inverse ? 1u : 0u})
        .threads(rows * (d / 2));
}

void dropout(const float* x, float* mask, float* y, size_t n, size_t units, uint64_t seed,
             uint64_t first_stream, float rate, float scale) {
    if (n == 0 || units == 0) return;
    const size_t blocks = (n + kDropBlock - 1) / kDropBlock;
    // The CPU's threshold, computed the CPU's way (activations.cpp).
    const auto threshold = static_cast<uint32_t>(rate * 65536.0f);
    const DropoutParams p{seed,        first_stream, u32(n), u32(units),
                          u32(blocks), threshold,    scale,  x ? 1u : 0u};
    const BufferRef jumps{ctx().dropout_jumps, 0};
    Launch launch(Kernel::DropoutFwd);
    launch.buf(x ? x : mask).buf(mask).buf(y ? y : mask);
    launch.raw_buf(jumps).params(p).threads(units * blocks * 256);
}

void adamw(float* w, const float* g, float* m, float* v, size_t n, const AdamW& p) {
    const AdamParams params{u32(n),    p.beta1, p.beta2, p.inv_bc1,
                            p.inv_bc2, p.lr,    p.eps,   p.weight_decay};
    Launch(Kernel::AdamW).buf(w).buf(g).buf(m).buf(v).params(params).threads(n);
}

void grad_norm(std::span<const Span> grads, float max_norm, float* out) {
    size_t chunks = 0;
    for (const Span& g : grads) chunks += (g.n + kNormChunk - 1) / kNormChunk;
    const Scratch part(chunks > 0 ? chunks : 1);
    size_t offset = 0;
    for (const Span& g : grads) {
        const size_t c = (g.n + kNormChunk - 1) / kNormChunk;
        if (c == 0) continue;
        Launch(Kernel::SumsqPartial)
            .buf(g.data)
            .buf(part.get())
            .params(SumsqParams{u32(g.n), u32(offset)})
            .groups(MTLSizeMake(c, 1, 1));
        offset += c;
    }
    if (offset == 0) fill(part.get(), 1, 0.0f);
    Launch(Kernel::NormFinish)
        .buf(part.get())
        .buf(out)
        .params(NormParams{u32(offset > 0 ? offset : 1), max_norm})
        .groups(MTLSizeMake(1, 1, 1));
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
