#include "transformer/tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <cstring>

void print_matrix(const std::string& name, const Tensor& t) {
    auto shape = t.getShape();
    if (shape.size() != 2) {
        std::cout << name << ": can only print 2D tensors\n";
        return;
    }

    std::cout << name << " [" << shape[0] << "x" << shape[1] << "]:\n";

    // Copy to CPU if needed
    float* host_data = new float[t.numel()];
    if (t.getDevice() == Device::CUDA) {
        cudaMemcpy(host_data, t.data(), t.numel() * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        std::memcpy(host_data, t.data(), t.numel() * sizeof(float));
    }

    for (int i = 0; i < shape[0]; i++) {
        std::cout << "  ";
        for (int j = 0; j < shape[1]; j++) {
            printf("%7.3f ", host_data[i * shape[1] + j]);
        }
        std::cout << "\n";
    }
    delete[] host_data;
}

bool verify_result(const Tensor& result, const float* expected, float tolerance = 1e-4) {
    float* host_data = new float[result.numel()];
    if (result.getDevice() == Device::CUDA) {
        cudaMemcpy(host_data, result.data(), result.numel() * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        std::memcpy(host_data, result.data(), result.numel() * sizeof(float));
    }

    bool all_correct = true;
    for (size_t i = 0; i < result.numel(); i++) {
        float diff = std::abs(host_data[i] - expected[i]);
        if (diff > tolerance) {
            std::cout << "  ❌ Mismatch at index " << i << ": got " << host_data[i]
                      << ", expected " << expected[i] << " (diff: " << diff << ")\n";
            all_correct = false;
        }
    }
    delete[] host_data;
    return all_correct;
}

void set_values(Tensor& t, const float* values) {
    if (t.getDevice() == Device::CUDA) {
        cudaMemcpy(t.data(), values, t.numel() * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        std::memcpy(t.data(), values, t.numel() * sizeof(float));
    }
}

int main() {
    std::cout << "=== cuBLAS Matrix Multiplication Test ===\n\n";

    // Test 1: Simple 2x2 matmul
    std::cout << "[Test 1] 2x2 Matrix Multiplication\n";
    {
        Tensor A({2, 2}, Device::CUDA);
        Tensor B({2, 2}, Device::CUDA);

        // A = [[1, 2],
        //      [3, 4]]
        float a_vals[] = {1, 2, 3, 4};
        set_values(A, a_vals);

        // B = [[5, 6],
        //      [7, 8]]
        float b_vals[] = {5, 6, 7, 8};
        set_values(B, b_vals);

        Tensor C = A.matmul(B);

        // Expected: [[19, 22],
        //            [43, 50]]
        // Calculation:
        //   C[0,0] = 1*5 + 2*7 = 19
        //   C[0,1] = 1*6 + 2*8 = 22
        //   C[1,0] = 3*5 + 4*7 = 43
        //   C[1,1] = 3*6 + 4*8 = 50

        float expected[] = {19, 22, 43, 50};

        print_matrix("A", A);
        print_matrix("B", B);
        print_matrix("C = A @ B", C);

        if (verify_result(C, expected)) {
            std::cout << "  ✓ Test 1 PASSED\n\n";
        } else {
            std::cout << "  ❌ Test 1 FAILED\n\n";
            return 1;
        }
    }

    // Test 2: Non-square matmul
    std::cout << "[Test 2] Non-Square Matrix Multiplication [3x2] @ [2x4]\n";
    {
        Tensor A({3, 2}, Device::CUDA);
        Tensor B({2, 4}, Device::CUDA);

        // A = [[1, 2],
        //      [3, 4],
        //      [5, 6]]
        float a_vals[] = {1, 2, 3, 4, 5, 6};
        set_values(A, a_vals);

        // B = [[1, 0, 0, 0],
        //      [0, 1, 0, 0]]
        float b_vals[] = {1, 0, 0, 0, 0, 1, 0, 0};
        set_values(B, b_vals);

        Tensor C = A.matmul(B);

        // Expected: [[1, 2, 0, 0],
        //            [3, 4, 0, 0],
        //            [5, 6, 0, 0]]
        float expected[] = {1, 2, 0, 0, 3, 4, 0, 0, 5, 6, 0, 0};

        print_matrix("A", A);
        print_matrix("B", B);
        print_matrix("C = A @ B", C);

        if (verify_result(C, expected)) {
            std::cout << "  ✓ Test 2 PASSED\n\n";
        } else {
            std::cout << "  ❌ Test 2 FAILED\n\n";
            return 1;
        }
    }

    // Test 3: Batched matmul (3D tensors)
    std::cout << "[Test 3] Batched Matrix Multiplication [2, 2, 2] @ [2, 2, 2]\n";
    {
        Tensor A({2, 2, 2}, Device::CUDA);
        Tensor B({2, 2, 2}, Device::CUDA);

        // Batch 0: [[1, 2],    Batch 1: [[5, 6],
        //           [3, 4]]              [7, 8]]
        float a_vals[] = {1, 2, 3, 4, 5, 6, 7, 8};
        set_values(A, a_vals);

        // Identity for both batches
        float b_vals[] = {1, 0, 0, 1, 1, 0, 0, 1};
        set_values(B, b_vals);

        Tensor C = A.matmul(B);

        // Expected: same as A (identity multiplication)
        float expected[] = {1, 2, 3, 4, 5, 6, 7, 8};

        if (verify_result(C, expected)) {
            std::cout << "  ✓ Test 3 PASSED (Batched multiplication works!)\n\n";
        } else {
            std::cout << "  ❌ Test 3 FAILED\n\n";
            return 1;
        }
    }

    // Test 4: Performance benchmark
    std::cout << "[Test 4] Performance Benchmark - Large Matrix (1024x1024)\n";
    {
        const int N = 1024;
        Tensor A({N, N}, Device::CUDA);
        Tensor B({N, N}, Device::CUDA);

        A.fill(1.0f);
        B.fill(2.0f);

        // Warm-up
        Tensor C_warmup = A.matmul(B);

        // Benchmark
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        Tensor C = A.matmul(B);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Calculate GFLOPS
        // For NxN @ NxN: 2*N^3 FLOPs (N^3 multiplies + N^3 adds)
        double flops = 2.0 * N * N * N;
        double gflops = (flops / 1e9) / (milliseconds / 1000.0);

        std::cout << "  Matrix size: " << N << "x" << N << "\n";
        std::cout << "  Time: " << milliseconds << " ms\n";
        std::cout << "  Performance: " << gflops << " GFLOPS\n";
        std::cout << "  ✓ Test 4 PASSED\n\n";

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Verify one element (should be N*1*2 = 2N)
        float host_val;
        cudaMemcpy(&host_val, C.data(), sizeof(float), cudaMemcpyDeviceToHost);
        float expected_val = N * 1.0f * 2.0f;

        if (std::abs(host_val - expected_val) < 1e-3) {
            std::cout << "  ✓ Correctness verified: C[0,0] = " << host_val
                      << " (expected " << expected_val << ")\n\n";
        } else {
            std::cout << "  ❌ Correctness FAILED: C[0,0] = " << host_val
                      << " (expected " << expected_val << ")\n\n";
            return 1;
        }
    }

    std::cout << "=== ALL MATMUL TESTS PASSED ===\n\n";
    std::cout << "Performance Notes:\n";
    std::cout << "  • Modern GPUs (RTX 3080): ~15-30 TFLOPS for FP32\n";
    std::cout << "  • A100 GPU: ~19.5 TFLOPS for FP32\n";
    std::cout << "  • If you got >1 TFLOPS, you're using the GPU properly!\n";
    std::cout << "  • If you got <100 GFLOPS, something is wrong (CPU fallback?)\n\n";
    std::cout << "Next Steps:\n";
    std::cout << "  1. ✓ Matmul works!\n";
    std::cout << "  2. Add elementwise kernels (add, multiply, GELU)\n";
    std::cout << "  3. Add softmax kernel\n";
    std::cout << "  4. Port attention to CUDA\n";

    return 0;
}
