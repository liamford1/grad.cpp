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
        int m = this->shape[this->shape.size() - 2];
        int k = this->shape.back();
        int n = other.shape.back();

        int batch_size = 1;
        if (shape.size() > 2) {
            for(size_t i = 0; i < shape.size() - 2; i++) batch_size *= shape[i];
        }

        float* c_ptr = result.data();
        const float* a_ptr = data();
        const float* b_ptr = other.data();

        int strideA = m * k;
        int strideB = k * n;
        int strideC = m * n;

        for(int b = 0; b < batch_size; b++) {
            const float* a_batch = a_ptr + b * strideA;
            const float* b_batch = b_ptr + b * strideB;
            float* c_batch = c_ptr + b * strideC;

            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    float sum = 0.0f;
                    for (int l = 0; l < k; ++l) {
                        sum += a_batch[i * k + l] * b_batch[l * n + j];
                    }
                    c_batch[i * n + j] = sum;
                }
            }
        }
        return result;
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

__global__ void add_inplace_kernel(float* a, const float* b, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] += b[idx];
    }
}

void Tensor::add_inplace(const Tensor& other) {
    if (numel() == other.numel() && shape == other.shape) {
        if (getDevice() == Device::CPU) {
            float* a_ptr = data();
            const float* b_ptr = other.data();
            for(size_t i=0; i<numel(); i++) a_ptr[i] += b_ptr[i];
        } else {
            int threads = 256;
            int blocks = (numel() + threads - 1) / threads;
            add_inplace_kernel<<<blocks, threads>>>(data(), other.data(), numel());
            cudaDeviceSynchronize();
        }
    } else if (!getIs3D() && other.getIs3D() &&
               getRows() == other.getRows() && getCols() == other.getCols()) {
        int batch_size = other.getBatchSize();
        int rows = getRows();
        int cols = getCols();

        float* dst = data();
        const float* src = other.data();

        if (getDevice() == Device::CPU) {
            for(int b = 0; b < batch_size; b++) {
                for(int i = 0; i < rows; i++) {
                    for(int j = 0; j < cols; j++) {
                        dst[i * cols + j] += src[b * rows * cols + i * cols + j];
                    }
                }
            }
        } else {
            for(int b = 0; b < batch_size; b++) {
                int threads = 256;
                int blocks = (rows * cols + threads - 1) / threads;
                add_inplace_kernel<<<blocks, threads>>>(dst, src + b * rows * cols, rows * cols);
                cudaDeviceSynchronize();
            }
        }
    } else if (!getIs3D() && !other.getIs3D()) {
        int my_rows = getRows();
        int my_cols = getCols();
        int other_rows = other.getRows();
        int other_cols = other.getCols();

        bool broadcast_rows = (other_rows == 1 && my_rows > 1);
        bool broadcast_cols = (other_cols == 1 && my_cols > 1);

        if ((broadcast_rows || other_rows == my_rows) &&
            (broadcast_cols || other_cols == my_cols)) {

            float* dst = data();
            const float* src = other.data();

            if (getDevice() == Device::CPU) {
                for(int i = 0; i < my_rows; i++) {
                    int src_i = broadcast_rows ? 0 : i;
                    for(int j = 0; j < my_cols; j++) {
                        int src_j = broadcast_cols ? 0 : j;
                        dst[i * my_cols + j] += src[src_i * other_cols + src_j];
                    }
                }
            } else {
                for(int i = 0; i < my_rows; i++) {
                    int src_i = broadcast_rows ? 0 : i;
                    for(int j = 0; j < my_cols; j++) {
                        int src_j = broadcast_cols ? 0 : j;
                        dst[i * my_cols + j] += src[src_i * other_cols + src_j];
                    }
                }
                cudaDeviceSynchronize();
            }
        } else {
            throw std::invalid_argument("Size mismatch in add_inplace: incompatible shapes for broadcasting");
        }
    } else {
        throw std::invalid_argument("Size mismatch in add_inplace: unsupported tensor configuration");
    }
}

void Tensor::xavier(size_t fan_in, size_t fan_out) {
    float scale = std::sqrt(2.0f / (fan_in + fan_out));

    std::vector<float> temp_data(numel());
    for(size_t i=0; i<numel(); i++) {
        temp_data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    }

    if (getDevice() == Device::CPU) {
        std::memcpy(data(), temp_data.data(), numel() * sizeof(float));
    } else {
        cudaMemcpy(data(), temp_data.data(), numel() * sizeof(float), cudaMemcpyHostToDevice);
    }
}

__global__ void softmax_kernel(const float* input, float* output, int rows, int cols) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx < rows) {
        const float* row_in = input + row_idx * cols;
        float* row_out = output + row_idx * cols;

        float max_val = -1e30f;
        for (int j = 0; j < cols; ++j) {
            if (row_in[j] > max_val) max_val = row_in[j];
        }

        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            float val = expf(row_in[j] - max_val);
            row_out[j] = val;
            sum += val;
        }

        for (int j = 0; j < cols; ++j) {
            row_out[j] /= sum;
        }
    }
}

Tensor Tensor::softmax() const {
    Tensor result(shape, getDevice());

    if (getDevice() == Device::CPU) {
        size_t size = numel();
        std::vector<float> h_data(size);
        std::memcpy(h_data.data(), data(), size * sizeof(float));

        int rows = size / shape.back();
        int cols = shape.back();

        for(int i=0; i<rows; i++) {
            float max_val = -1e9;
            for(int j=0; j<cols; j++) max_val = std::max(max_val, h_data[i*cols + j]);
            float sum = 0.0f;
            for(int j=0; j<cols; j++) {
                h_data[i*cols + j] = std::exp(h_data[i*cols + j] - max_val);
                sum += h_data[i*cols + j];
            }
            for(int j=0; j<cols; j++) h_data[i*cols + j] /= sum;
        }
        std::memcpy(result.data(), h_data.data(), size * sizeof(float));
    }
    else {
        size_t cols = shape.back();
        size_t rows = numel() / cols;

        int threads = 256;
        int blocks = (rows + threads - 1) / threads;

        softmax_kernel<<<blocks, threads>>>(data(), result.data(), rows, cols);
        cudaDeviceSynchronize();
    }
    return result;
}

__global__ void log_softmax_kernel(const float* input, float* output, int rows, int cols) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx < rows) {
        const float* row_in = input + row_idx * cols;
        float* row_out = output + row_idx * cols;

        float max_val = -1e30f;
        for (int j = 0; j < cols; ++j) {
            if (row_in[j] > max_val) max_val = row_in[j];
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < cols; ++j) {
            sum_exp += expf(row_in[j] - max_val);
        }
        float log_sum = logf(sum_exp);

        for (int j = 0; j < cols; ++j) {
            row_out[j] = (row_in[j] - max_val) - log_sum;
        }
    }
}

Tensor Tensor::log_softmax() const {
    Tensor result(shape, getDevice());

    if (getDevice() == Device::CPU) {
        size_t size = numel();
        size_t cols = shape.back();
        size_t rows = size / cols;

        const float* in_ptr = data();
        float* out_ptr = result.data();

        for(int i=0; i<rows; i++) {
            const float* row_in = in_ptr + i*cols;
            float* row_out = out_ptr + i*cols;

            float max_val = -1e9;
            for(int j=0; j<cols; j++) if(row_in[j] > max_val) max_val = row_in[j];

            float sum = 0.0f;
            for(int j=0; j<cols; j++) sum += std::exp(row_in[j] - max_val);
            float log_sum = std::log(sum);

            for(int j=0; j<cols; j++) row_out[j] = (row_in[j] - max_val) - log_sum;
        }
    }
    else {
        size_t cols = shape.back();
        size_t rows = numel() / cols;
        int threads = 256;
        int blocks = (rows + threads - 1) / threads;

        log_softmax_kernel<<<blocks, threads>>>(data(), result.data(), rows, cols);
        cudaDeviceSynchronize();
    }
    return result;
}

Tensor Tensor::create_causal_mask(size_t seq_len) {
    std::vector<int> shape = { (int)seq_len, (int)seq_len };
    Tensor mask(shape, Device::CPU);

    float* ptr = mask.data();
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < seq_len; j++) {
            if (j > i) ptr[i * seq_len + j] = -1e9f;
            else ptr[i * seq_len + j] = 0.0f;
        }
    }

    return mask;
}

float Tensor::getValue(int b, int r, int c) const {
    int idx = 0;
    if (shape.size() == 3) {
        idx = b * strides[0] + r * strides[1] + c * strides[2];
    } else {
        idx = b * (shape[1] * shape[2]) + r * shape[2] + c;
    }

    float h_val;
    if (getDevice() == Device::CPU) {
        return data()[idx];
    } else {
        cudaMemcpy(&h_val, data() + idx, sizeof(float), cudaMemcpyDeviceToHost);
        return h_val;
    }
}

void Tensor::setValue(int b, int r, int c, float value) {
    int idx = 0;
    if (shape.size() == 3) {
        idx = b * strides[0] + r * strides[1] + c * strides[2];
    } else {
        idx = b * (shape[1] * shape[2]) + r * shape[2] + c;
    }

    if (getDevice() == Device::CPU) {
        data()[idx] = value;
    } else {
        cudaMemcpy(data() + idx, &value, sizeof(float), cudaMemcpyHostToDevice);
    }
}

__global__ void scale_kernel(float* data, float factor, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= factor;
    }
}

Tensor Tensor::scale(float factor) const {
    Tensor result = this->clone();

    if (getDevice() == Device::CPU) {
        float* ptr = result.data();
        for(size_t i=0; i<numel(); i++) ptr[i] *= factor;
    } else {
        int threads = 256;
        int blocks = (numel() + threads - 1) / threads;
        scale_kernel<<<blocks, threads>>>(result.data(), factor, numel());
        cudaDeviceSynchronize();
    }
    return result;
}

Tensor Tensor::slice(size_t r_start, size_t r_end, size_t c_start, size_t c_end) const {
    size_t new_rows = r_end - r_start;
    size_t new_cols = c_end - c_start;

    Tensor result(new_rows, new_cols, getDevice());

    size_t src_cols = shape.back();
    size_t dst_cols = new_cols;
    size_t width_bytes = new_cols * sizeof(float);

    size_t src_offset_idx = r_start * src_cols + c_start;

    if (getDevice() == Device::CPU) {
        for(size_t i=0; i<new_rows; i++) {
            float* dst_row = result.data() + i * dst_cols;
            const float* src_row = data() + src_offset_idx + i * src_cols;
            std::memcpy(dst_row, src_row, width_bytes);
        }
    } else {
        const float* src_ptr = data() + src_offset_idx;
        float* dst_ptr = result.data();

        size_t spitch = src_cols * sizeof(float);
        size_t dpitch = dst_cols * sizeof(float);

        cudaMemcpy2D(dst_ptr, dpitch, src_ptr, spitch, width_bytes, new_rows, cudaMemcpyDeviceToDevice);
    }

    return result;
}

__global__ void elementwise_mul_kernel(float* a, const float* b, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] *= b[idx];
    }
}

Tensor Tensor::elementwise(const Tensor& other) const {
    Tensor result = this->clone();

    if (getDevice() == Device::CPU) {
        float* a_ptr = result.data();
        const float* b_ptr = other.data();
        for(size_t i=0; i<numel(); i++) a_ptr[i] *= b_ptr[i];
    } else {
        int threads = 256;
        int blocks = (numel() + threads - 1) / threads;
        elementwise_mul_kernel<<<blocks, threads>>>(result.data(), other.data(), numel());
        cudaDeviceSynchronize();
    }
    return result;
}

Tensor Tensor::subtract(const Tensor& other) const {
    Tensor neg_other = other.scale(-1.0f);
    Tensor result = this->add(neg_other);
    return result;
}

__global__ void gelu_kernel(const float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float cube = x * x * x;
        output[idx] = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * cube)));
    }
}

Tensor Tensor::gelu() const {
    Tensor result(shape, getDevice());
    size_t size = numel();

    if (getDevice() == Device::CPU) {
        const float* in_ptr = data();
        float* out_ptr = result.data();
        for(size_t i = 0; i < size; i++) {
            float x = in_ptr[i];
            float cube = x * x * x;
            out_ptr[i] = 0.5f * x * (1.0f + std::tanh(0.79788456f * (x + 0.044715f * cube)));
        }
    } else {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        gelu_kernel<<<blocks, threads>>>(data(), result.data(), size);
        cudaDeviceSynchronize();
    }
    return result;
}
