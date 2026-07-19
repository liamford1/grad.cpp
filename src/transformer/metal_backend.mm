#include "transformer/metal_backend.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstdint>
#include <cstdlib>
#include <mutex>

namespace metalgpu {
namespace {

constexpr size_t kPageBytes = 16384;

bool page_aligned(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) % kPageBytes) == 0;
}

size_t round_up_to_page(size_t n) {
    return ((n + kPageBytes - 1) / kPageBytes) * kPageBytes;
}

struct Context {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    bool ok = false;

    Context() {
        if (const char* env = std::getenv("TRANSFORMER_METAL")) {
            if (env[0] == '0') return;
        }
        device = MTLCreateSystemDefaultDevice();
        if (!device) return;
        // Unified memory is the whole premise: without it we would need
        // real transfers and a very different design.
        if (![device hasUnifiedMemory]) return;
        queue = [device newCommandQueue];
        ok = (queue != nil);
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

MPSMatrix* make_matrix(id<MTLBuffer> buf, int rows, int cols) {
    MPSMatrixDescriptor* desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                              columns:cols
                                             rowBytes:cols * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:buf descriptor:desc];
}

}  // namespace

bool available() {
    return ctx().ok;
}

bool sgemm(const float* A, const float* B, float* C,
           int M, int N, int K, bool transA, bool transB,
           float alpha, float beta) {
    Context& c = ctx();
    if (!c.ok) return false;
    if (!page_aligned(A) || !page_aligned(B) || !page_aligned(C)) return false;

    const int a_rows = transA ? K : M;
    const int a_cols = transA ? M : K;
    const int b_rows = transB ? N : K;
    const int b_cols = transB ? K : N;

    @autoreleasepool {
        std::lock_guard<std::mutex> lk(gpu_mutex);

        id<MTLBuffer> bufA = wrap(c.device, A, static_cast<size_t>(a_rows) * a_cols * sizeof(float));
        id<MTLBuffer> bufB = wrap(c.device, B, static_cast<size_t>(b_rows) * b_cols * sizeof(float));
        id<MTLBuffer> bufC = wrap(c.device, C, static_cast<size_t>(M) * N * sizeof(float));
        if (!bufA || !bufB || !bufC) return false;

        MPSMatrixMultiplication* mm =
            [[MPSMatrixMultiplication alloc] initWithDevice:c.device
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

        [mm encodeToCommandBuffer:cb
                       leftMatrix:make_matrix(bufA, a_rows, a_cols)
                      rightMatrix:make_matrix(bufB, b_rows, b_cols)
                     resultMatrix:make_matrix(bufC, M, N)];
        [cb commit];
        [cb waitUntilCompleted];

        return cb.status == MTLCommandBufferStatusCompleted;
    }
}

}  // namespace metalgpu
