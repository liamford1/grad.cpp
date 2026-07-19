#include "transformer/tensor.h"
#include "transformer/blas_wrapper.h"
#include "transformer/parallel.h"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <random>
#include <stdexcept>
#include <utility>
#include <cassert>
#include <string>

namespace {

// Large tensor storage is page-aligned and page-rounded. On Apple Silicon
// the CPU and GPU share physical memory, so Metal can wrap these pages in
// an MTLBuffer directly (newBufferWithBytesNoCopy) - the GPU computes on
// the same bytes the CPU sees, no transfer. That API requires page
// alignment and page-multiple lengths; 16KB covers both Apple Silicon
// (16K pages) and Linux (4K pages).
//
// Only tensors big enough to plausibly hit the GPU matmul path get the
// aligned treatment: page-aligned allocation bypasses malloc's small-block
// cache and touches fresh kernel pages, which measurably slowed training
// when applied to every scratch tensor. Small tensors use plain new[];
// the Metal backend's alignment check routes them to the CPU automatically.
constexpr size_t kPageBytes = 16384;
constexpr size_t kAlignThresholdBytes = 256 * 1024;

// The element count decides which allocator was used, so free recomputes
// the same rule. Tensor dimensions are immutable after construction, which
// makes this sound.
float* alloc_floats(size_t n) {
    const size_t raw_bytes = n * sizeof(float);
    if (raw_bytes >= kAlignThresholdBytes) {
        size_t bytes = ((raw_bytes + kPageBytes - 1) / kPageBytes) * kPageBytes;
        void* p = nullptr;
        if (posix_memalign(&p, kPageBytes, bytes) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<float*>(p);
    }
    return new float[n];
}

void free_floats(float* p, size_t n) {
    if (!p) return;
    if (n * sizeof(float) >= kAlignThresholdBytes) {
        std::free(p);
    } else {
        delete[] p;
    }
}

}  // namespace

Tensor::Tensor() {
    this->rows = 0;
    this->cols = 0;
    this->batch_size = 0;
    this->is_3d = false;
    this->data = nullptr;
}

Tensor::Tensor(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Tensor dimensions must be positive");
    }

    this->rows = rows;
    this->cols = cols;
    this->batch_size = 1;
    this->is_3d = false;

    size_t total = batch_size * rows * cols;

    if (total > MAX_TENSOR_ELEMENTS) {
        throw std::overflow_error("Tensor too large: " + std::to_string(total) +
                                  " elements exceeds maximum of " + std::to_string(MAX_TENSOR_ELEMENTS));
    }

    this->data = alloc_floats(total);
    std::memset(this->data, 0, total * sizeof(float));
}

Tensor::Tensor(size_t batch_size, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0 || batch_size == 0) {
        throw std::invalid_argument("Tensor dimensions must be positive");
    }

    this->rows = rows;
    this->cols = cols;
    this->batch_size = batch_size;
    this->is_3d = true;

    size_t total = batch_size * rows * cols;

    if (total > MAX_TENSOR_ELEMENTS) {
        throw std::overflow_error("Tensor too large: " + std::to_string(total) +
                                  " elements exceeds maximum of " + std::to_string(MAX_TENSOR_ELEMENTS));
    }

    this->data = alloc_floats(total);
    std::memset(this->data, 0, total * sizeof(float));
}

Tensor Tensor::uninitialized(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Tensor dimensions must be positive");
    }
    Tensor t;
    t.rows = rows;
    t.cols = cols;
    t.batch_size = 1;
    t.is_3d = false;
    size_t total = rows * cols;
    if (total > MAX_TENSOR_ELEMENTS) {
        throw std::overflow_error("Tensor too large: " + std::to_string(total) +
                                  " elements exceeds maximum of " + std::to_string(MAX_TENSOR_ELEMENTS));
    }
    t.data = alloc_floats(total);
    return t;
}

Tensor Tensor::uninitialized(size_t batch_size, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0 || batch_size == 0) {
        throw std::invalid_argument("Tensor dimensions must be positive");
    }
    Tensor t;
    t.rows = rows;
    t.cols = cols;
    t.batch_size = batch_size;
    t.is_3d = true;
    size_t total = batch_size * rows * cols;
    if (total > MAX_TENSOR_ELEMENTS) {
        throw std::overflow_error("Tensor too large: " + std::to_string(total) +
                                  " elements exceeds maximum of " + std::to_string(MAX_TENSOR_ELEMENTS));
    }
    t.data = alloc_floats(total);
    return t;
}

Tensor::Tensor(const Tensor& other) {
    this->rows = other.rows;
    this->cols = other.cols;
    this->batch_size = other.batch_size;
    this->is_3d = other.is_3d;

    size_t total = batch_size * rows * cols;

    if (total > MAX_TENSOR_ELEMENTS) {
        throw std::overflow_error("Tensor too large: " + std::to_string(total) +
                                  " elements exceeds maximum of " + std::to_string(MAX_TENSOR_ELEMENTS));
    }

    this->data = alloc_floats(total);
    std::memcpy(this->data, other.data, total * sizeof(float));
}

Tensor::Tensor(Tensor&& other) noexcept :
    data(other.data),
    rows(other.rows),
    cols(other.cols),
    batch_size(other.batch_size),
    is_3d(other.is_3d) 
{
    other.data = nullptr;
    other.rows = other.cols = other.batch_size = 0;
    other.is_3d = false;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;
    Tensor tmp(other);
    
    std::swap(data, tmp.data);
    std::swap(rows, tmp.rows);
    std::swap(cols, tmp.cols);
    std::swap(batch_size, tmp.batch_size);
    std::swap(is_3d, tmp.is_3d);
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;
    free_floats(data, batch_size * rows * cols);
    data = other.data;
    rows = other.rows;
    cols = other.cols;
    batch_size = other.batch_size;
    is_3d = other.is_3d;
    other.data = nullptr;
    other.rows = other.cols = other.batch_size = 0;
    other.is_3d = false;
    return *this;
}

Tensor::~Tensor() {
    free_floats(data, batch_size * rows * cols);
}

//2D Tensor methods
float Tensor::getValue(size_t row, size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Tensor(2D) index out of bounds");
    }
    return data[row * cols + col];
}

void Tensor::setValue(size_t row, size_t col, float value) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Tensor(2D) index out of bounds");
    }
    data[row * cols + col] = value;
}

//3D Tensor methods
float Tensor::getValue(size_t batch, size_t row, size_t col) const {
    if (batch >= batch_size || row >= rows || col >= cols) {
        throw std::out_of_range("Tensor(3D) index out of bounds");
    }
    return data[batch * rows * cols + row * cols + col];
}

void Tensor::setValue(size_t batch, size_t row, size_t col, float value) {
    if (batch >= batch_size || row >= rows || col >= cols) {
        throw std::out_of_range("Tensor(3D) index out of bounds");
    }
    data[batch * rows * cols + row * cols + col] = value;
}

void Tensor::display() const {
    assertValid("display(this)");
    if (is_3d) {
        std::cout << "[display] showing batch 0 of " << batch_size << "\n";
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                std::cout << getValue(0, i, j) << " ";
            }
            std::cout << "\n";
        }
        return;
    }
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) std::cout << getValue(i, j) << " ";
        std::cout << "\n";
    }
}

Tensor Tensor::matmul(const Tensor& other) const {
    assertValid("matmul(lhs)");
    other.assertValid("matmul(rhs)");

    if (!this->is_3d && !other.is_3d) {
        if (this->cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication");
        }

        const size_t M = this->rows;
        const size_t K = this->cols;
        const size_t N = other.cols;

        Tensor result = Tensor::uninitialized(M, N);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K,
                    1.0f,
                    this->data, K,
                    other.data, N,
                    0.0f,
                    result.data, N);

        return result;
        
    } else if (this->is_3d && !other.is_3d) {
        if (this->cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions do not match for batch multiplication");
        }

        const size_t batch_count = this->batch_size;
        const size_t M = this->rows;
        const size_t K = this->cols;
        const size_t N = other.cols;

        Tensor result = Tensor::uninitialized(batch_count, M, N);

        for (size_t b = 0; b < batch_count; ++b) {
            const float* A = this->data + b * M * K;
            float* C = result.data + b * M * N;
            
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K,
                        1.0f,
                        A, K,
                        other.data, N,
                        0.0f,
                        C, N);
        }
        
        return result;
        
    } else if (this->is_3d && other.is_3d) {
        if (this->batch_size != other.batch_size || this->cols != other.rows) {
            throw std::invalid_argument("Batch matrix dimensions do not match");
        }

        const size_t batch_count = this->batch_size;
        const size_t M = this->rows;
        const size_t K = this->cols;
        const size_t N = other.cols;

        Tensor result = Tensor::uninitialized(batch_count, M, N);

        for (size_t b = 0; b < batch_count; ++b) {
            const float* A = this->data + b * M * K;
            const float* B = other.data + b * K * N;
            float* C = result.data + b * M * N;
            
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K,
                        1.0f,
                        A, K,
                        B, N,
                        0.0f,
                        C, N);
        }
        
        return result;
        
    } else {
        throw std::invalid_argument("Unsupported matrix multiplication configuration");
    }
}

Tensor Tensor::add(const Tensor& other) const {
    assertValid("add(lhs)");
    other.assertValid("add(rhs)");
    if (!this->is_3d && !other.is_3d) {
        bool rows_compatible = (rows == other.rows) || (rows == 1) || (other.rows == 1);
        bool cols_compatible = (cols == other.cols) || (cols == 1) || (other.cols == 1);
        if (!rows_compatible || !cols_compatible) {
            throw std::invalid_argument("Shapes not broadcastable");
        }

        const size_t R = std::max(rows, other.rows);
        const size_t C = std::max(cols, other.cols);
        Tensor result = Tensor::uninitialized(R, C);

        const float* A = this->data;
        const float* B = other.data;
        float* Out = result.raw();

        const bool a_row_bcast = (rows == 1);
        const bool a_col_bcast = (cols == 1);
        const bool b_row_bcast = (other.rows == 1);
        const bool b_col_bcast = (other.cols == 1);

        for (size_t i = 0; i < R; ++i) {
            const size_t ai = a_row_bcast ? 0 : i;
            const size_t bi = b_row_bcast ? 0 : i;
            const size_t a_row_off = ai * cols;
            const size_t b_row_off = bi * other.cols;
            const size_t out_row_off = i * C;

            for (size_t j = 0; j < C; ++j) {
                const size_t aj = a_col_bcast ? 0 : j;
                const size_t bj = b_col_bcast ? 0 : j;
                Out[out_row_off + j] = A[a_row_off + aj] + B[b_row_off + bj];
            }
        }
        return result;
    } else if (this->is_3d && !other.is_3d) {
        bool rows_compatible = (rows == other.rows) || (rows == 1) || (other.rows == 1);
        bool cols_compatible = (cols == other.cols) || (cols == 1) || (other.cols == 1);
        if (!rows_compatible || !cols_compatible) {
            throw std::invalid_argument("Tensor dimensions don't match for broadcasting");
        }

        Tensor result = Tensor::uninitialized(batch_size, rows, cols);
        const float* A = this->raw();
        const float* B = other.raw();
        float* C = result.raw();
        
        const bool b_row_bcast = (other.rows == 1);
        const bool b_col_bcast = (other.cols == 1);

        for (size_t b = 0; b < batch_size; ++b) {
            const size_t batch_offset = b * rows * cols;
            for (size_t i = 0; i < rows; ++i) {
                const size_t oi = b_row_bcast ? 0 : i;
                const size_t row_offset = batch_offset + i * cols;
                const size_t other_row_offset = oi * other.cols;

                for (size_t j = 0; j < cols; ++j) {
                    const size_t oj = b_col_bcast ? 0 : j;
                    C[row_offset + j] = A[row_offset + j] + B[other_row_offset + oj];
                }
            }
        }
        return result;
        
    } else if (!this->is_3d && other.is_3d) {
        bool rows_compatible = (this->rows == other.rows) || (this->rows == 1) || (other.rows == 1);
        bool cols_compatible = (this->cols == other.cols) || (this->cols == 1) || (other.cols == 1);
        if (!rows_compatible || !cols_compatible) {
            throw std::invalid_argument("Tensor dimensions don't match for broadcasting");
        }

        Tensor result = Tensor::uninitialized(other.batch_size, other.rows, other.cols);
        const float* A = this->raw();
        const float* B = other.raw();
        float* C = result.raw();
        
        const bool a_row_bcast = (this->rows == 1);
        const bool a_col_bcast = (this->cols == 1);

        for (size_t b = 0; b < other.batch_size; b++) {
            const size_t batch_offset = b * other.rows * other.cols;
            for (size_t i = 0; i < other.rows; i++) {
                const size_t this_i = a_row_bcast ? 0 : i;
                const size_t row_offset = batch_offset + i * other.cols;
                const size_t this_row_offset = this_i * this->cols;

                for (size_t j = 0; j < other.cols; j++) {
                    const size_t this_j = a_col_bcast ? 0 : j;
                    C[row_offset + j] = A[this_row_offset + this_j] + B[row_offset + j];
                }
            }
        }
        return result;

    } else if (this->is_3d && other.is_3d) {
        if (this->batch_size != other.batch_size || this->rows != other.rows || this->cols != other.cols) {
            throw std::invalid_argument("3D tensor dimensions don't match");
        }

        Tensor result = Tensor::uninitialized(this->batch_size, this->rows, this->cols);
        const float* A = this->raw();
        const float* B = other.raw();
        float* C = result.raw();
        const size_t total = this->batch_size * this->rows * this->cols;

        blas_vadd(A, B, C, total);
        return result;
    } else {
        throw std::invalid_argument("Unsupported addition configuration");
    }
}

Tensor Tensor::subtract(const Tensor& other) const {
    assertValid("subtract(lhs)");
    other.assertValid("subtract(rhs)");

    if (!this->is_3d && !other.is_3d) {
        if (this->rows != other.rows || this->cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions do not match for subtraction");
        }
        Tensor result = Tensor::uninitialized(this->rows, this->cols);
        const size_t total = this->rows * this->cols;

        blas_vsub(data, other.data, result.data, total);
        
        return result;
    } else if (this->is_3d && other.is_3d) {
        if (this->batch_size != other.batch_size || this->rows != other.rows || this->cols != other.cols) {
            throw std::invalid_argument("3D tensor dimensions don't match for subtraction");
        }
        Tensor result = Tensor::uninitialized(this->batch_size, this->rows, this->cols);
        const size_t total = this->batch_size * this->rows * this->cols;

        blas_vsub(data, other.data, result.data, total);

        return result;
    } else {
        throw std::invalid_argument("Cannot subtract tensors with different dimensionalities");
    }
}

Tensor Tensor::elementwise(const Tensor& other) const {
    assertValid("elementwise(lhs)");
    other.assertValid("elementwise(rhs)");
    
    if (!this->is_3d && !other.is_3d) {
        if (this->rows != other.rows || this->cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions do not match for elementwise multiply");
        }
        Tensor result = Tensor::uninitialized(this->rows, this->cols);
        const size_t total = this->rows * this->cols;

        blas_vmul(data, other.data, result.data, total);
        
        return result;
        
    } else if (this->is_3d && other.is_3d) {
        if (this->batch_size != other.batch_size || this->rows != other.rows || this->cols != other.cols) {
            throw std::invalid_argument("3D tensor dimensions don't match for elementwise multiply");
        }
        Tensor result = Tensor::uninitialized(this->batch_size, this->rows, this->cols);
        const size_t total = this->batch_size * this->rows * this->cols;

        blas_vmul(data, other.data, result.data, total);
        return result;
    } else {
        throw std::invalid_argument("Cannot perform elementwise multiply on tensors with different dimensionalities");
    }
}

Tensor Tensor::transpose() const {
    assertValid("transpose(this)");
    
    if (!this->is_3d) {
        Tensor result = Tensor::uninitialized(this->cols, this->rows);
        const float* src = this->raw();
        float* dst = result.raw();

        const size_t BLOCK = 32;

        for (size_t i0 = 0; i0 < this->rows; i0 += BLOCK) {
            const size_t i_max = std::min(i0 + BLOCK, this->rows);
            for (size_t j0 = 0; j0 < this->cols; j0 += BLOCK) {
                const size_t j_max = std::min(j0 + BLOCK, this->cols);

                for (size_t i = i0; i < i_max; ++i) {
                    for (size_t j = j0; j < j_max; ++j) {
                        dst[j * this->rows + i] = src[i * this->cols + j];
                    }
                }
            }
        }
        return result;

    } else {
        Tensor result = Tensor::uninitialized(this->batch_size, this->cols, this->rows);
        const float* src = this->raw();
        float* dst = result.raw();

        const size_t BLOCK = 32;

        for (size_t b = 0; b < this->batch_size; ++b) {
            const float* batch_src = src + b * this->rows * this->cols;
            float* batch_dst = dst + b * this->cols * this->rows;

            for (size_t i0 = 0; i0 < this->rows; i0 += BLOCK) {
                const size_t i_max = std::min(i0 + BLOCK, this->rows);
                for (size_t j0 = 0; j0 < this->cols; j0 += BLOCK) {
                    const size_t j_max = std::min(j0 + BLOCK, this->cols);

                    for (size_t i = i0; i < i_max; ++i) {
                        for (size_t j = j0; j < j_max; ++j) {
                            batch_dst[j * this->rows + i] = batch_src[i * this->cols + j];
                        }
                    }
                }
            }
        }
        return result;
    }
}

Tensor Tensor::softmax() const {
    assertValid("softmax(this)");

    // Rows are contiguous for both 2D and 3D tensors, so a single loop over
    // batch*rows covers both. exp goes through vec_exp (SIMD).
    const size_t total_rows = this->is_3d ? this->batch_size * this->rows
                                          : this->rows;
    Tensor result = this->is_3d
        ? Tensor::uninitialized(this->batch_size, this->rows, this->cols)
        : Tensor::uninitialized(this->rows, this->cols);
    const float* input_data = this->raw();
    float* output_data = result.raw();
    const size_t cols = this->cols;

    parallel_for(total_rows, 16, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; i++) {
            const float* row_in = input_data + i * cols;
            float* row_out = output_data + i * cols;

            float max_val = row_in[0];
            for (size_t j = 1; j < cols; j++) {
                max_val = std::max(max_val, row_in[j]);
            }
            for (size_t j = 0; j < cols; j++) {
                row_out[j] = row_in[j] - max_val;
            }
            vec_exp(row_out, row_out, static_cast<int>(cols));

            const float inv_sum = 1.0f / vec_sum(row_out, static_cast<int>(cols));
            for (size_t j = 0; j < cols; j++) {
                row_out[j] *= inv_sum;
            }
        }
    });
    return result;
}

void Tensor::fill(float value) {
    assertValid("fill(this)");
    const size_t total = batch_size * rows * cols;
    blas_vfill(value, data, total);
}

Tensor Tensor::scale(float scaler) const {
    assertValid("scale(this)");

    if (!this->is_3d) {
        Tensor result = Tensor::uninitialized(this->rows, this->cols);
        const float* src = this->raw();
        float* dst = result.raw();
        const size_t total = this->rows * this->cols;

        blas_vsmul(src, scaler, dst, total);
        return result;
    } else {
        Tensor result = Tensor::uninitialized(this->batch_size, this->rows, this->cols);
        const float* src = this->raw();
        float* dst = result.raw();
        const size_t total = this->batch_size * this->rows * this->cols;

        blas_vsmul(src, scaler, dst, total);
        return result;
    }
}

Tensor Tensor::reshape(size_t new_rows, size_t new_cols) const {
    assertValid("reshape(this)");
    if (is_3d) throw std::invalid_argument("reshape: 3D not supported yet");
    if (new_rows * new_cols != rows * cols) {
        throw std::invalid_argument("Matrix sizes do not match for reshape");
    }
    Tensor result = Tensor::uninitialized(new_rows, new_cols);
    float* out = result.raw();
    const float* in = data;
    for (size_t i = 0; i < new_rows * new_cols; ++i) out[i] = in[i];
    return result;
}

Tensor Tensor::slice(size_t start_row, size_t num_rows, size_t start_col, size_t num_cols) const {
    assertValid("slice(this)");
    if (is_3d) throw std::invalid_argument("slice: 3D not supported yet");

    if (start_row + num_rows > this->rows || start_col + num_cols > this->cols) {
        throw std::invalid_argument("Out of bounds error");
    }

    Tensor result(num_rows, num_cols);

    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < num_cols; j++) {
            result.setValue(i, j, this->getValue(start_row + i, start_col + j));
        }
    }

    return result;
}

Tensor Tensor::concatenate(const Tensor& other, int axis) const {
    assertValid("concatenate(lhs)");
    other.assertValid("concatenate(rhs)");
    if (is_3d || other.is_3d) {
        throw std::invalid_argument("concatenate: 3D not supported yet");
    }
    if (axis == 0 && this->cols != other.cols) {
        throw std::invalid_argument("Columns do not match for axis=0 concatenation");
    } 
    if (axis == 1 && this->rows != other.rows) {
        throw std::invalid_argument("Rows do not match for axis=1 concatenation");
    }
    if (axis != 0 && axis != 1) {
        throw std::invalid_argument("Invalid axis: must be 0 or 1");
    }

    if (axis == 0) {
        Tensor result(this->rows + other.rows, this->cols);

        for (size_t i = 0; i < this->rows; i++) {
            for (size_t j = 0; j < this->cols; j++) {
                result.setValue(i, j, this->getValue(i, j));
            }
        }

        for (size_t i = 0; i < other.rows; i++) {
            for (size_t j = 0; j < other.cols; j++) {
                result.setValue(this->rows + i, j, other.getValue(i, j));
            }
        }

        return result;
    } else {
        Tensor result(this->rows, this->cols + other.cols);

        for (size_t i = 0; i < this->rows; i++) {
            for (size_t j = 0; j < this->cols; j++) {
                result.setValue(i, j, this->getValue(i, j));
            }
        }

        for (size_t i = 0; i < other.rows; i++) {
            for (size_t j = 0; j < other.cols; j++) {
                result.setValue(i, this->cols + j, other.getValue(i, j));
            }
        }

        return result;
    }
}

void Tensor::xavier(size_t fan_in, size_t fan_out) {
    assertValid("xavier(target)");
    static std::random_device rd;
    static std::mt19937 gen(rd());

    float limit = std::sqrt(6.0f / (fan_in + fan_out));
    std::uniform_real_distribution<float> dis(-limit, limit);

    size_t total = batch_size * rows * cols;
    for (size_t i = 0; i < total; i++) {
        data[i] = dis(gen);
    }
}

Tensor Tensor::create_causal_mask(size_t seq_len) {
    Tensor mask = Tensor::uninitialized(seq_len, seq_len);
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < seq_len; j++) {
            if (j > i) {
                mask.setValue(i, j, -1e9f);
            } else {
                mask.setValue(i, j, 0.0f);
            }
        }
    }
    return mask;
}

Tensor Tensor::create_causal_mask_batch(size_t batch_size, size_t seq_len) {
    Tensor mask = Tensor::uninitialized(batch_size, seq_len, seq_len);
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < seq_len; j++) {
                if (j > i) {
                    mask.setValue(b, i, j, -1e9f);
                } else {
                    mask.setValue(b, i, j, 0.0f);
                }
            }
        }
    }
    return mask;
}

void Tensor::assertValid(const std::string& context) const {
    if (data == nullptr) {
        throw std::runtime_error("Tensor error [" + context + "]: data pointer is null");
    }
    if (rows == 0 || cols == 0) {
        throw std::runtime_error("Tensor error [" + context + "]: invalid shape (" +
                                 std::to_string(rows) + "x" + std::to_string(cols) + ")");
    }
    if (is_3d && batch_size == 0) {
        throw std::runtime_error("Tensor error [" + context + "]: invalid batch_size " +
                                 std::to_string(batch_size));
    }
}

void Tensor::scale_inplace(float scalar) {
    assertValid("scale_inplace");
    const size_t total = batch_size * rows * cols;
    blas_vsmul(data, scalar, data, total);
}

void Tensor::add_inplace(const Tensor& other) {
    assertValid("add_inplace");
    other.assertValid("add_inplace(other)");
    
    if (!is_3d && !other.is_3d) {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Shape mismatch for in-place add");
        }

        const size_t total = rows * cols;
        const float* other_data = other.raw();
        blas_vadd(data, other_data, data, total);

    } else if (is_3d && other.is_3d) {
        if (batch_size != other.batch_size || rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Shape mismatch for in-place add");
        }

        const size_t total = batch_size * rows * cols;
        const float* other_data = other.raw();
        blas_vadd(data, other_data, data, total);

    } else {
        throw std::invalid_argument("Cannot add 2D and 3D tensors in-place");
    }
}

void Tensor::multiply_inplace(const Tensor& other) {
    assertValid("multiply_inplace");
    other.assertValid("multiply_inplace(other)");

    if (rows != other.rows || cols != other.cols || is_3d != other.is_3d) {
        throw std::invalid_argument("Shape mismatch for in-place multiply");
    }

    const size_t total = (is_3d ? batch_size : 1) * rows * cols;
    const float* other_data = other.raw();
    blas_vmul(data, other_data, data, total);

}

void Tensor::zero() {
    const size_t total = (is_3d ? batch_size : 1) * rows * cols;
    std::memset(data, 0, total * sizeof(float));
}