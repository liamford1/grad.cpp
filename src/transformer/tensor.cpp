#include "transformer/tensor.h"
#include "transformer/blas_wrapper.h"
#include "transformer/parallel.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <random>
#include <stdexcept>
#include <utility>
#include <cassert>
#include <string>
#include <type_traits>

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

void check_dims(size_t batch_size, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0 || batch_size == 0) {
        throw std::invalid_argument("Tensor dimensions must be positive");
    }
    // Compared by division so the check cannot itself overflow: a product
    // that wraps size_t would otherwise pass as a small tensor.
    if (cols > MAX_TENSOR_ELEMENTS / rows ||
        batch_size > MAX_TENSOR_ELEMENTS / (rows * cols)) {
        throw std::overflow_error("Tensor too large: " + std::to_string(batch_size) + "x" +
                                  std::to_string(rows) + "x" + std::to_string(cols) +
                                  " exceeds the maximum of " +
                                  std::to_string(MAX_TENSOR_ELEMENTS) + " elements");
    }
}

// Parameter initialization stream. Guarded because nothing stops two
// threads from constructing models at once; xavier runs once per parameter
// at construction, so the lock is never on a hot path. A function-local
// static so a Tensor built during another file's static initialization
// still finds it constructed.
struct InitStream {
    std::mutex mutex;
    std::mt19937 gen;  // default-constructed: std::mt19937's fixed seed 5489
};

InitStream& init_stream() {
    static InitStream s;
    return s;
}

}  // namespace

// std::vector<Tensor> and the optimizer's moment maps only move (rather
// than deep-copy) on reallocation if these hold; assert them instead of
// trusting that a future member never quietly breaks them.
static_assert(std::is_nothrow_default_constructible_v<Tensor>);
static_assert(std::is_nothrow_move_constructible_v<Tensor>);
static_assert(std::is_nothrow_move_assignable_v<Tensor>);
static_assert(std::is_nothrow_swappable_v<Tensor>);

// The allocator choice is made exactly once, here, and stamped into the
// deleter that travels with the pointer. There is no second copy of the
// size rule for the release path to drift away from.
Tensor::Storage Tensor::alloc_floats(size_t n) {
    const size_t raw_bytes = n * sizeof(float);
    if (raw_bytes >= kAlignThresholdBytes) {
        const size_t bytes = ((raw_bytes + kPageBytes - 1) / kPageBytes) * kPageBytes;
        void* p = nullptr;
        if (posix_memalign(&p, kPageBytes, bytes) != 0) {
            throw std::bad_alloc();
        }
        return Storage(static_cast<float*>(p), Deleter{true});
    }
    return Storage(new float[n], Deleter{false});
}

Tensor::Tensor(size_t rows, size_t cols)
    : rows(rows), cols(cols), batch_size(1), is_3d(false) {
    check_dims(1, rows, cols);
    data = alloc_floats(numel());
    std::memset(data.get(), 0, numel() * sizeof(float));
}

Tensor::Tensor(size_t batch_size, size_t rows, size_t cols)
    : rows(rows), cols(cols), batch_size(batch_size), is_3d(true) {
    check_dims(batch_size, rows, cols);
    data = alloc_floats(numel());
    std::memset(data.get(), 0, numel() * sizeof(float));
}

Tensor Tensor::uninitialized(size_t rows, size_t cols) {
    check_dims(1, rows, cols);
    Tensor t;
    t.rows = rows; t.cols = cols; t.batch_size = 1; t.is_3d = false;
    t.data = alloc_floats(t.numel());
    return t;
}

Tensor Tensor::uninitialized(size_t batch_size, size_t rows, size_t cols) {
    check_dims(batch_size, rows, cols);
    Tensor t;
    t.rows = rows; t.cols = cols; t.batch_size = batch_size; t.is_3d = true;
    t.data = alloc_floats(t.numel());
    return t;
}

Tensor::Tensor(const Tensor& other)
    : rows(other.rows), cols(other.cols), batch_size(other.batch_size), is_3d(other.is_3d) {
    if (const size_t total = numel(); total > 0) {
        data = alloc_floats(total);
        std::memcpy(data.get(), other.data.get(), total * sizeof(float));
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : data(std::move(other.data)),
      rows(std::exchange(other.rows, 0)),
      cols(std::exchange(other.cols, 0)),
      batch_size(std::exchange(other.batch_size, 0)),
      is_3d(std::exchange(other.is_3d, false)) {}

// Copy-and-swap: the copy may throw, but *this is not touched until it has
// succeeded, so assignment is strongly exception-safe. The temporary takes
// the old storage with it when it dies.
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) Tensor(other).swap(*this);
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) Tensor(std::move(other)).swap(*this);
    return *this;
}

void Tensor::swap(Tensor& other) noexcept {
    using std::swap;
    swap(data, other.data);
    swap(rows, other.rows);
    swap(cols, other.cols);
    swap(batch_size, other.batch_size);
    swap(is_3d, other.is_3d);
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
                    this->data.get(), K,
                    other.data.get(), N,
                    0.0f,
                    result.data.get(), N);

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
            const float* A = this->data.get() + b * M * K;
            float* C = result.data.get() + b * M * N;
            
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K,
                        1.0f,
                        A, K,
                        other.data.get(), N,
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
            const float* A = this->data.get() + b * M * K;
            const float* B = other.data.get() + b * K * N;
            float* C = result.data.get() + b * M * N;
            
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

        const float* A = this->data.get();
        const float* B = other.data.get();
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
        // The result takes this tensor's shape, so only the 2D operand may
        // broadcast (a size-1 row or column dimension).
        const bool rows_compatible = (other.rows == rows) || (other.rows == 1);
        const bool cols_compatible = (other.cols == cols) || (other.cols == 1);
        if (!rows_compatible || !cols_compatible) {
            throw std::invalid_argument("add: 2D operand does not broadcast to the 3D shape");
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
        // Mirror of the case above: the result takes other's shape.
        const bool rows_compatible = (this->rows == other.rows) || (this->rows == 1);
        const bool cols_compatible = (this->cols == other.cols) || (this->cols == 1);
        if (!rows_compatible || !cols_compatible) {
            throw std::invalid_argument("add: 2D operand does not broadcast to the 3D shape");
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

        blas_vsub(data.get(), other.data.get(), result.data.get(), total);
        
        return result;
    } else if (this->is_3d && other.is_3d) {
        if (this->batch_size != other.batch_size || this->rows != other.rows || this->cols != other.cols) {
            throw std::invalid_argument("3D tensor dimensions don't match for subtraction");
        }
        Tensor result = Tensor::uninitialized(this->batch_size, this->rows, this->cols);
        const size_t total = this->batch_size * this->rows * this->cols;

        blas_vsub(data.get(), other.data.get(), result.data.get(), total);

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

        blas_vmul(data.get(), other.data.get(), result.data.get(), total);
        
        return result;
        
    } else if (this->is_3d && other.is_3d) {
        if (this->batch_size != other.batch_size || this->rows != other.rows || this->cols != other.cols) {
            throw std::invalid_argument("3D tensor dimensions don't match for elementwise multiply");
        }
        Tensor result = Tensor::uninitialized(this->batch_size, this->rows, this->cols);
        const size_t total = this->batch_size * this->rows * this->cols;

        blas_vmul(data.get(), other.data.get(), result.data.get(), total);
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
    blas_vfill(value, data.get(), total);
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

Tensor Tensor::slice(size_t start_row, size_t num_rows, size_t start_col, size_t num_cols) const {
    assertValid("slice(this)");
    if (is_3d) throw std::invalid_argument("slice: 2D tensors only");

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

void Tensor::xavier(size_t fan_in, size_t fan_out) {
    assertValid("xavier(target)");

    float limit = std::sqrt(6.0f / (fan_in + fan_out));
    std::uniform_real_distribution<float> dis(-limit, limit);

    InitStream& stream = init_stream();
    std::lock_guard<std::mutex> lk(stream.mutex);
    const size_t total = numel();
    for (size_t i = 0; i < total; i++) {
        data[i] = dis(stream.gen);
    }
}

void Tensor::set_init_seed(uint64_t seed) {
    InitStream& stream = init_stream();
    std::lock_guard<std::mutex> lk(stream.mutex);
    // mt19937 takes a 32-bit seed; fold the high half in rather than drop it.
    stream.gen.seed(static_cast<std::mt19937::result_type>(seed ^ (seed >> 32)));
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

void Tensor::assertValid(const std::string& context) const {
    if (!data.get()) {
        throw std::runtime_error("Tensor error [" + context + "]: data.get() pointer is null");
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
    blas_vsmul(data.get(), scalar, data.get(), total);
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
        blas_vadd(data.get(), other_data, data.get(), total);

    } else if (is_3d && other.is_3d) {
        if (batch_size != other.batch_size || rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Shape mismatch for in-place add");
        }

        const size_t total = batch_size * rows * cols;
        const float* other_data = other.raw();
        blas_vadd(data.get(), other_data, data.get(), total);

    } else {
        throw std::invalid_argument("Cannot add 2D and 3D tensors in-place");
    }
}

void Tensor::multiply_inplace(const Tensor& other) {
    assertValid("multiply_inplace");
    other.assertValid("multiply_inplace(other)");

    if (rows != other.rows || cols != other.cols || is_3d != other.is_3d ||
        (is_3d && batch_size != other.batch_size)) {
        throw std::invalid_argument("Shape mismatch for in-place multiply");
    }

    blas_vmul(data.get(), other.raw(), data.get(), numel());

}

void Tensor::zero() {
    const size_t total = (is_3d ? batch_size : 1) * rows * cols;
    std::memset(data.get(), 0, total * sizeof(float));
}