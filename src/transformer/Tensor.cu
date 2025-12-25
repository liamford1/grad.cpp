#include "transformer/tensor.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstring>
#include <iostream>

// ------------------------------------------------------------
// CUBLAS SINGLETON (The "Handle")
// Industry Standard: Create the handle once, reuse it forever.
// ------------------------------------------------------------
class CublasContext {
private:
    cublasHandle_t handle;
    CublasContext() {
        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS handle");
        }
    }
    ~CublasContext() {
        cublasDestroy(handle);
    }
public:
    static cublasHandle_t getHandle() {
        static CublasContext instance;
        return instance.handle;
    }
};

// ------------------------------------------------------------
// CUDA KERNELS (The "GPU Code")
// ------------------------------------------------------------

__global__ void fill_kernel(float* data, size_t size, float value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

// ------------------------------------------------------------
// TENSOR IMPLEMENTATION
// ------------------------------------------------------------

Tensor::Tensor(std::vector<int> shape, Device device) {
    // 1. Setup Metadata
    this->shape = shape;
    this->strides = calculate_contiguous_strides(shape);
    this->storage_offset = 0;

    // 2. Calculate Total Elements
    size_t total_elements = 1;
    for (int s : shape) total_elements *= s;

    // 3. Allocate Storage
    // This calls your Storage constructor (malloc or cudaMalloc)
    this->storage = std::make_shared<Storage>(total_elements * sizeof(float), device);

    // 4. Initialize Data (Safe default: 0.0f)
    // We cannot just loop here because the data might be on GPU!
    this->fill(0.0f);
}

void Tensor::fill(float value) {
    size_t size = numel();

    if (storage->device == Device::CPU) {
        // CPU: Standard C++ fill
        float* ptr = data();
        std::fill(ptr, ptr + size, value);
    }
    else {
        // GPU: Launch CUDA Kernel
        float* ptr = data(); // This returns a DEVICE pointer

        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        fill_kernel<<<blocks, threads>>>(ptr, size, value);

        // Check for launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA Kernel Failed: " + std::string(cudaGetErrorString(err)));
        }

        // Wait for GPU to finish (optional, but good for debugging)
        cudaDeviceSynchronize();
    }
}

// Deep Copy
Tensor Tensor::clone() const {
    Tensor copy(this->shape, this->getDevice());

    // We need a copy kernel eventually, but for now let's reuse cudaMemcpy
    size_t bytes = numel() * sizeof(float);

    if (getDevice() == Device::CPU) {
        std::memcpy(copy.data(), this->data(), bytes);
    } else {
        cudaMemcpy(copy.data(), this->data(), bytes, cudaMemcpyDeviceToDevice);
    }

    return copy;
}

// ------------------------------------------------------------
// MATRIX MULTIPLICATION (The Workhorse)
// ------------------------------------------------------------
Tensor Tensor::matmul(const Tensor& other) const {
    // 1. Shape Checks
    if (this->shape.size() < 2 || other.shape.size() < 2) {
        throw std::invalid_argument("matmul requires at least 2D tensors");
    }

    if (this->shape.back() != other.shape[other.shape.size() - 2]) {
        throw std::invalid_argument("Matmul shape mismatch: "
            + std::to_string(this->shape.back()) + " vs "
            + std::to_string(other.shape[other.shape.size() - 2]));
    }

    // 2. Setup Output Tensor
    std::vector<int> result_shape = this->shape;
    result_shape.back() = other.shape.back(); // [M, K] * [K, N] -> [M, N]

    Tensor result(result_shape, this->getDevice());

    // 3. Dispatch to Device
    if (this->getDevice() == Device::CPU) {
        // Fallback to CPU implementation (use BLAS or warn)
        throw std::runtime_error("CPU Matmul not implemented in Tensor.cu yet! Use CUDA tensors.");
    }
    else {
        // 4. CUDA Implementation (cuBLAS)
        // We perform C = A * B

        // For 2D case: [M, K] x [K, N] -> [M, N]
        int m = this->shape[this->shape.size() - 2];  // rows of A
        int k = this->shape.back();                    // cols of A (rows of B)
        int n = other.shape.back();                    // cols of B

        // Handle batch dimensions if present
        int batch_size = 1;
        if (shape.size() > 2) {
            for(size_t i = 0; i < shape.size() - 2; i++) {
                batch_size *= shape[i];
            }
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;

        if (batch_size == 1) {
            // Simple 2D matmul
            // NOTE: cuBLAS is Column-Major. C++ is Row-Major.
            // To compute C = A * B in Row-Major, we actually compute:
            // C^T = B^T * A^T
            // We swap A and B in the call and tell cuBLAS they are "Normal" (non-transposed).
            // This calculates B * A in Col-Major, which is exactly A * B in Row-Major.

            cublasStatus_t status = cublasSgemm(
                CublasContext::getHandle(),
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,           // dimensions flipped: n, m instead of m, n
                &alpha,
                other.data(), n,   // B comes first! ldb = n
                this->data(), k,   // A comes second! lda = k
                &beta,
                result.data(), n   // C Result, ldc = n
            );

            if (status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("cuBLAS Sgemm Failed with status: " + std::to_string(status));
            }
        } else {
            // Batched matmul (for 3D+ tensors)
            long long int strideA = m * k;
            long long int strideB = k * n;
            long long int strideC = m * n;

            cublasStatus_t status = cublasSgemmStridedBatched(
                CublasContext::getHandle(),
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                other.data(), n, strideB,
                this->data(), k, strideA,
                &beta,
                result.data(), n, strideC,
                batch_size
            );

            if (status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("cuBLAS SgemmStridedBatched Failed with status: " + std::to_string(status));
            }
        }

        // Synchronize to ensure computation is complete
        cudaDeviceSynchronize();
    }

    return result;
}

float Tensor::getValue(int r, int c) const {
    int idx = 0;
    if (shape.size() == 2) {
        idx = r * strides[0] + c * strides[1];
    } else {
        idx = r * (shape.size() > 1 ? shape[1] : 1) + c;
    }

    float h_val;
    if (getDevice() == Device::CPU) {
        return data()[idx];
    } else {
        cudaMemcpy(&h_val, data() + idx, sizeof(float), cudaMemcpyDeviceToHost);
        return h_val;
    }
}

void Tensor::setValue(int r, int c, float value) {
    int idx = 0;
    if (shape.size() == 2) {
        idx = r * strides[0] + c * strides[1];
    } else {
        idx = r * (shape.size() > 1 ? shape[1] : 1) + c;
    }

    if (getDevice() == Device::CPU) {
        data()[idx] = value;
    } else {
        cudaMemcpy(data() + idx, &value, sizeof(float), cudaMemcpyHostToDevice);
    }
}
