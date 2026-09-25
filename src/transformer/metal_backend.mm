#include "grad/transformer/metal_backend.h"
#include "grad/utils/env.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

namespace grad::metal {
namespace {

constexpr size_t kPageBytes = 16384;

bool page_aligned(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) % kPageBytes) == 0;
}

size_t round_up_to_page(size_t n) {
    return ((n + kPageBytes - 1) / kPageBytes) * kPageBytes;
}

// fp32 -> fp16 conversion kernel, run on the GPU so operand halving costs
// one bandwidth-bound pass instead of a CPU round trip. Compiled from
// source at startup; if compilation fails the backend silently stays fp32.
constexpr const char* kConvertSource = R"(
#include <metal_stdlib>
using namespace metal;
kernel void f32_to_f16(device const float* src [[buffer(0)]],
                       device half* dst [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    dst[i] = half(src[i]);
}
)";

struct Context {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> convert = nil;
    // Persistent GPU-private scratch for the fp16 operand copies, grown as
    // needed and reused across calls (guarded by gpu_mutex).
    id<MTLBuffer> scratch[2] = {nil, nil};
    size_t scratch_cap[2] = {0, 0};
    bool fp16 = false;
    bool ok = false;

    Context() {
        if (const char* value = env::lookup("GRAD_METAL", "TRANSFORMER_METAL")) {
            if (value[0] == '0') return;
        }
        device = MTLCreateSystemDefaultDevice();
        if (!device) return;
        // Unified memory is the whole premise: without it we would need
        // real transfers and a very different design.
        if (![device hasUnifiedMemory]) return;
        queue = [device newCommandQueue];
        ok = (queue != nil);
        if (!ok) return;

        // Off by default: measured at the current model scale, fp16
        // operands neither help (logits matmuls: bandwidth win cancels
        // conversion cost) nor let smaller matmuls win on the GPU (FFN at
        // a 5 GFLOP threshold ran 14% slower - synchronous dispatch still
        // loses to AMX). See BENCHMARKS.md #11. The machinery stays for
        // the next model size up, where the GPU's margin grows.
        bool want_fp16 = false;
        if (const char* value = env::lookup("GRAD_METAL_FP16", "TRANSFORMER_METAL_FP16")) {
            if (value[0] == '1') want_fp16 = true;
        }
        if (want_fp16) {
            NSError* error = nil;
            id<MTLLibrary> lib = [device newLibraryWithSource:@(kConvertSource)
                                                      options:nil
                                                        error:&error];
            id<MTLFunction> fn = lib ? [lib newFunctionWithName:@"f32_to_f16"] : nil;
            convert = fn ? [device newComputePipelineStateWithFunction:fn error:&error] : nil;
            fp16 = (convert != nil);
        }
    }
};

Context& ctx() {
    static Context c;
    return c;
}

// One GPU job at a time. Calls from inside parallel_for regions are small
// (below the FLOP threshold) and never reach here.
std::mutex gpu_mutex;

// Wraps existing page-aligned memory in an MTLBuffer without copying.
// Tensor storage is page-rounded (see tensor.cpp), so the rounded length
// stays within the allocation.
id<MTLBuffer> wrap(id<MTLDevice> device, const void* p, size_t bytes) {
    return [device newBufferWithBytesNoCopy:const_cast<void*>(p)
                                     length:round_up_to_page(bytes)
                                    options:MTLResourceStorageModeShared
                                deallocator:nil];
}

MPSMatrix* make_matrix(id<MTLBuffer> buf, size_t rows, size_t cols, MPSDataType dtype) {
    const size_t elem = (dtype == MPSDataTypeFloat16) ? 2 : sizeof(float);
    MPSMatrixDescriptor* desc = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                                      columns:cols
                                                                     rowBytes:cols * elem
                                                                     dataType:dtype];
    return [[MPSMatrix alloc] initWithBuffer:buf descriptor:desc];
}

id<MTLBuffer> scratch_fp16(Context& c, int slot, size_t elements) {
    const size_t bytes = elements * 2;
    if (c.scratch_cap[slot] < bytes) {
        c.scratch[slot] = [c.device newBufferWithLength:bytes
                                                options:MTLResourceStorageModePrivate];
        c.scratch_cap[slot] = c.scratch[slot] ? bytes : 0;
    }
    return c.scratch[slot];
}

}  // namespace

bool available() {
    return ctx().ok;
}

bool fp16_active() {
    return ctx().ok && ctx().fp16;
}

bool sgemm(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, bool transA,
           bool transB, float alpha, float beta) {
    Context& c = ctx();
    if (!c.ok) return false;
    if (!page_aligned(A) || !page_aligned(B) || !page_aligned(C)) return false;

    const size_t a_rows = transA ? K : M;
    const size_t a_cols = transA ? M : K;
    const size_t b_rows = transB ? N : K;
    const size_t b_cols = transB ? K : N;
    const size_t nA = a_rows * a_cols;
    const size_t nB = b_rows * b_cols;

    // Every early return below happens before commit, when nothing has
    // been submitted and C is untouched, so the caller's CPU fallback is
    // safe. Once committed, the GPU may have written part of C (and with
    // beta != 0 consumed its old contents), so a failure there cannot be
    // retried and is thrown instead - outside the autorelease pool, which
    // is not drained by C++ unwinding.
    std::string failure;
    @autoreleasepool {
        std::lock_guard<std::mutex> lk(gpu_mutex);

        id<MTLBuffer> bufA = wrap(c.device, A, nA * sizeof(float));
        id<MTLBuffer> bufB = wrap(c.device, B, nB * sizeof(float));
        id<MTLBuffer> bufC = wrap(c.device, C, M * N * sizeof(float));
        if (!bufA || !bufB || !bufC) return false;

        id<MTLBuffer> halfA = nil;
        id<MTLBuffer> halfB = nil;
        if (c.fp16) {
            halfA = scratch_fp16(c, 0, nA);
            halfB = scratch_fp16(c, 1, nB);
        }
        const bool use_fp16 = (halfA != nil && halfB != nil);

        MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc] initWithDevice:c.device
                                                                        transposeLeft:transA
                                                                       transposeRight:transB
                                                                           resultRows:M
                                                                        resultColumns:N
                                                                      interiorColumns:K
                                                                                alpha:alpha
                                                                                 beta:beta];
        if (!mm) return false;

        id<MTLCommandBuffer> cb = [c.queue commandBuffer];
        if (!cb) return false;

        if (use_fp16) {
            // Halve the operands on-device, then multiply fp16 x fp16 with
            // an fp32 result matrix: MPS accumulates (and applies
            // alpha/beta) in full precision.
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            if (!enc) return false;  // messages to nil would skip the conversion
            [enc setComputePipelineState:c.convert];
            const NSUInteger tg = c.convert.maxTotalThreadsPerThreadgroup;
            [enc setBuffer:bufA offset:0 atIndex:0];
            [enc setBuffer:halfA offset:0 atIndex:1];
            [enc dispatchThreads:MTLSizeMake(nA, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg < 256 ? tg : 256, 1, 1)];
            [enc setBuffer:bufB offset:0 atIndex:0];
            [enc setBuffer:halfB offset:0 atIndex:1];
            [enc dispatchThreads:MTLSizeMake(nB, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg < 256 ? tg : 256, 1, 1)];
            [enc endEncoding];

            [mm encodeToCommandBuffer:cb
                           leftMatrix:make_matrix(halfA, a_rows, a_cols, MPSDataTypeFloat16)
                          rightMatrix:make_matrix(halfB, b_rows, b_cols, MPSDataTypeFloat16)
                         resultMatrix:make_matrix(bufC, M, N, MPSDataTypeFloat32)];
        } else {
            [mm encodeToCommandBuffer:cb
                           leftMatrix:make_matrix(bufA, a_rows, a_cols, MPSDataTypeFloat32)
                          rightMatrix:make_matrix(bufB, b_rows, b_cols, MPSDataTypeFloat32)
                         resultMatrix:make_matrix(bufC, M, N, MPSDataTypeFloat32)];
        }
        [cb commit];
        [cb waitUntilCompleted];

        if (cb.status != MTLCommandBufferStatusCompleted) {
            NSError* err = cb.error;
            failure = err ? std::string([[err localizedDescription] UTF8String])
                          : std::string("status ") + std::to_string(static_cast<long>(cb.status));
        }
    }
    if (!failure.empty()) {
        throw std::runtime_error("Metal sgemm failed after submission (C may be partially "
                                 "written, so it cannot fall back to the CPU): "
                                 + failure);
    }
    return true;
}

}  // namespace grad::metal
