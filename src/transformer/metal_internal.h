#pragma once

// Objective-C++ internals of the Metal backend, shared by metal_backend.mm
// (device, allocator, stream, CPU-mode sgemm) and metal_ops.mm (kernels).
// Not a public header: it needs -fobjc-arc and the Metal frameworks.

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace grad::metal {

// The kernels of metal_kernels.metal that the resident stream launches,
// with the threadgroup width each requires (0 = any).
enum class Kernel : size_t {
    Fill,
    Copy,
    Add,
    Acc,
    AccScaled,
    Mul,
    MulAcc,
    Scale,
    ScaleBy,
    AddRows,
    BroadcastAdd,
    BroadcastReduce,
    ColPartial,
    ColFinish,
    GeluFwd,
    GeluBwd,
    SiluFwd,
    SiluBwd,
    LayerNormFwd,
    LayerNormBwdDx,
    LayerNormBwdParams,
    SoftmaxFwd,
    SoftmaxBwd,
    LogSoftmaxFwd,
    LogSoftmaxBwd,
    AttnSoftmaxFwd,
    AttnSoftmaxBwd,
    NllFwd,
    NllBwd,
    CeFwd,
    MeanReduce,
    CeBwd,
    EmbeddingFwd,
    EmbeddingBwd,
    Rope,
    DropoutFwd,
    AdamW,
    SumsqPartial,
    NormFinish,
    GemmBatched,
    Count,
};

struct KernelInfo {
    const char* name;
    size_t threads;
};

// Row kernels reduce over a fixed 256-wide tree (kRow in the .metal).
inline constexpr size_t kRowThreads = 256;

inline constexpr std::array<KernelInfo, static_cast<size_t>(Kernel::Count)> kKernels = {{
    {"ew_fill", 0},
    {"ew_copy", 0},
    {"ew_add", 0},
    {"ew_acc", 0},
    {"ew_acc_scaled", 0},
    {"ew_mul", 0},
    {"ew_mul_acc", 0},
    {"ew_scale", 0},
    {"ew_scale_by", 0},
    {"add_rows", 0},
    {"broadcast_add", 0},
    {"broadcast_reduce", 0},
    {"col_partial", 0},
    {"col_finish", 0},
    {"gelu_fwd", 0},
    {"gelu_bwd", 0},
    {"silu_fwd", 0},
    {"silu_bwd", 0},
    {"layer_norm_fwd", kRowThreads},
    {"layer_norm_bwd_dx", kRowThreads},
    {"layer_norm_bwd_params", 0},
    {"softmax_fwd", kRowThreads},
    {"softmax_bwd", kRowThreads},
    {"log_softmax_fwd", kRowThreads},
    {"log_softmax_bwd", kRowThreads},
    {"attn_softmax_fwd", kRowThreads},
    {"attn_softmax_bwd", kRowThreads},
    {"nll_fwd", kRowThreads},
    {"nll_bwd", 0},
    {"ce_fwd", kRowThreads},
    {"mean_reduce", kRowThreads},
    {"ce_bwd", 0},
    {"embedding_fwd", 0},
    {"embedding_bwd", 0},
    {"rope", 0},
    {"dropout_fwd", 0},
    {"adamw", 0},
    {"sumsq_partial", kRowThreads},
    {"norm_finish", kRowThreads},
    {"gemm_batched", 128},
}};

struct Context {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;

    // CPU-mode offload: the fp16 conversion kernel and its persistent
    // GPU-private scratch, grown as needed and reused across calls.
    id<MTLComputePipelineState> convert = nil;
    id<MTLBuffer> scratch[2] = {nil, nil};
    size_t scratch_cap[2] = {0, 0};
    bool fp16 = false;
    bool ok = false;

    // Resident mode, initialized on first use (init_resident).
    std::once_flag resident_once;
    bool resident_ok = false;
    std::string resident_status = "not initialized";
    std::array<id<MTLComputePipelineState>, static_cast<size_t>(Kernel::Count)> pipelines{};
    // Dropout jump-ahead matrices (metal_kernels.metal, dropout_fwd).
    id<MTLBuffer> dropout_jumps = nil;

    Context();
    void init_resident();
};

Context& ctx();

// A buffer and byte offset addressing some pool pointer.
struct BufferRef {
    id<MTLBuffer> buffer = nil;
    NSUInteger offset = 0;
};

// The resident allocator. Blocks are page-rounded shared MTLBuffers cached
// by size. A released block is reusable only once every command buffer
// that might reference it has completed: the release is tagged with the
// open command buffer's sequence number and reclaimed by that buffer's
// completion handler, or at the next sync.
class Pool {
public:
    float* allocate(size_t bytes);
    void release(const float* p) noexcept;
    // The block containing p; false when p is not pool memory.
    bool resolve(const void* p, BufferRef& out);
    void reclaim(uint64_t completed_seq);
    void reclaim_all();
    void empty_cache();
    void add_stats(size_t& live, size_t& peak, size_t& cached);

private:
    struct Block {
        id<MTLBuffer> buffer = nil;
        float* data = nullptr;  // [buffer contents]
        size_t size = 0;
        uint64_t tag = 0;
        bool live = false;
    };
    void make_free_locked(Block& block);

    std::mutex mu_;
    // By base address, ordered so resolve can find the block containing an
    // interior pointer (std::less orders unrelated pointers totally).
    std::map<const float*, Block, std::less<>> blocks_;
    std::unordered_map<size_t, std::vector<float*>> free_;
    std::vector<float*> deferred_;
    size_t live_bytes_ = 0;
    size_t peak_live_bytes_ = 0;
    size_t cached_bytes_ = 0;
    std::atomic<size_t> leaked_blocks_{0};
};

Pool& pool();

// The resident command stream: one open command buffer and, between MPS
// calls, one open compute encoder with serial dispatch, so each kernel sees
// its predecessor's writes. The buffer is committed every commit_every
// dispatches without waiting.
class Stream {
public:
    Stream();

    // Encodes one compute dispatch: fn(encoder) inside the open encoder.
    template <typename Fn>
    void compute(Fn&& fn) {
        std::lock_guard<std::mutex> lk(mu_);
        @autoreleasepool {
            if (!open_locked()) return fail_open();
            if (!enc_) enc_ = [cb_ computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
            if (!enc_) return fail_open();
            fn(enc_);
            dispatched_locked();
        }
    }

    // Encodes work that needs the command buffer itself (MPS): the open
    // compute encoder is ended first.
    template <typename Fn>
    void command(Fn&& fn) {
        std::lock_guard<std::mutex> lk(mu_);
        @autoreleasepool {
            if (!open_locked()) return fail_open();
            end_encoder_locked();
            fn(cb_);
            dispatched_locked();
        }
    }

    void fence();
    void synchronize();
    void flush();

    // No encoded work that has not completed.
    [[nodiscard]] bool idle() const;
    // The sequence number of the latest command buffer that may reference
    // memory touched so far.
    [[nodiscard]] uint64_t current_seq() const;

    void add_stats(StreamStats& out) const;

private:
    bool open_locked();
    void end_encoder_locked();
    void dispatched_locked();
    void commit_locked();
    void throttle_locked();
    // Records "could not create a command buffer or encoder" for the next
    // sync to report; the encode is dropped.
    void fail_open();
    std::string drain_errors_locked(bool all);

    mutable std::mutex mu_;
    id<MTLCommandBuffer> cb_ = nil;
    id<MTLComputeCommandEncoder> enc_ = nil;
    std::deque<id<MTLCommandBuffer>> in_flight_;
    std::string error_;
    size_t commit_every_ = 32;
    size_t max_in_flight_ = 16;
    size_t open_dispatches_ = 0;
    std::atomic<uint64_t> open_seq_{1};
    std::atomic<uint64_t> completed_seq_{0};
    std::atomic<bool> has_open_work_{false};
    std::atomic<bool> pending_{false};
    std::atomic<uint64_t> syncs_{0};
    std::atomic<uint64_t> command_buffers_{0};
    std::atomic<uint64_t> dispatches_{0};
    std::atomic<uint64_t> throttle_waits_{0};
};

Stream& stream();

}  // namespace grad::metal
