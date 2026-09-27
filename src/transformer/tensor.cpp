#include "grad/transformer/tensor.h"
#include "grad/transformer/blas_wrapper.h"
#include "grad/transformer/device.h"
#include "grad/transformer/metal_backend.h"
#include "grad/transformer/metal_ops.h"
#include "grad/transformer/parallel.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace grad {

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
constexpr size_t kAlignThresholdBytes = size_t{256} * 1024;

// In Metal mode a new pool block is idle by construction (the pool reuses a
// block only once the GPU is done with it), so the CPU may clear it with no
// fence. Up to this size memset is cheaper than encoding a fill kernel
// (~5 us); above it the encoding thread would spend its time in memset
// while the GPU waits for work (a 70M backward zero-fills ~1 GB of
// activation gradients per micro-batch), so the GPU clears it instead.
constexpr size_t kCpuZeroFillBytes = size_t{64} * 1024;

void check_dims(const Shape& shape) {
    if (shape.rank() == 0) {
        throw std::invalid_argument("Tensor needs at least one dimension");
    }
    size_t total = 1;
    for (size_t axis = 0; axis < shape.rank(); axis++) {
        const size_t n = shape[axis];
        if (n == 0) {
            throw std::invalid_argument("Tensor dimensions must be positive, got "
                                        + shape.to_string());
        }
        // Compared by division so the check cannot itself overflow: a
        // product that wraps size_t would otherwise pass as a small tensor.
        if (n > kMaxTensorElements / total) {
            throw std::overflow_error("Tensor too large: " + shape.to_string()
                                      + " exceeds the maximum of "
                                      + std::to_string(kMaxTensorElements) + " elements");
        }
        total *= n;
    }
}

// True when a and b have the same rank and agree on every dimension
// outside the innermost two, i.e. index the same set of matrices.
bool same_leading_dims(const Shape& a, const Shape& b) {
    if (a.rank() != b.rank()) return false;
    for (size_t axis = 0; axis + 2 < a.rank(); axis++) {
        if (a[axis] != b[axis]) return false;
    }
    return true;
}

void require_same_shape(const Tensor& a, const Tensor& b, const char* op) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument(std::string(op) + ": shapes " + a.shape().to_string() + " and "
                                    + b.shape().to_string() + " differ");
    }
}

// out = a + b where either operand may be a 2D tensor broadcast over the
// other. Each operand is walked as (batch, rows, cols) with a stride of 0
// along every dimension it broadcasts, so one loop covers every case.
struct BroadcastOperand {
    const float* data;
    size_t batch_stride, row_stride, col_stride;
};

BroadcastOperand broadcast_operand(const Tensor& t, size_t rows, size_t cols) {
    const bool batched = t.getIs3D();
    return {t.raw(), batched ? t.getRows() * t.getCols() : 0, t.getRows() == rows ? t.getCols() : 0,
            t.getCols() == cols ? size_t{1} : size_t{0}};
}

void broadcast_add(const BroadcastOperand& a, const BroadcastOperand& b, float* out, size_t batch,
                   size_t rows, size_t cols) {
    for (size_t n = 0; n < batch; n++) {
        for (size_t i = 0; i < rows; i++) {
            const float* ar = a.data + n * a.batch_stride + i * a.row_stride;
            const float* br = b.data + n * b.batch_stride + i * b.row_stride;
            float* o = out + (n * rows + i) * cols;
            if (a.col_stride == 1 && b.col_stride == 1) {
                for (size_t j = 0; j < cols; j++) o[j] = ar[j] + br[j];
            } else {
                for (size_t j = 0; j < cols; j++)
                    o[j] = ar[j * a.col_stride] + br[j * b.col_stride];
            }
        }
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

namespace tensor_detail {

void release_device_storage(const float* p) noexcept {
    metal::release(p);
}

void fence_device() {
    metal::fence();
}

}  // namespace tensor_detail

Shape::Shape(std::initializer_list<size_t> dims) {
    if (dims.size() > kMaxRank) {
        throw std::invalid_argument("Shape: rank " + std::to_string(dims.size())
                                    + " exceeds the maximum of " + std::to_string(kMaxRank));
    }
    numel_ = 1;
    for (size_t n : dims) {
        dims_[rank_++] = n;
        numel_ *= n;
    }
}

Shape Shape::with_last_dim(size_t n) const noexcept {
    Shape s = *this;
    s.dims_[s.rank_ - 1] = n;
    s.numel_ = 1;
    for (size_t axis = 0; axis < s.rank_; axis++) s.numel_ *= s.dims_[axis];
    return s;
}

Shape Shape::transposed() const noexcept {
    Shape s = *this;
    if (rank_ > 1) std::swap(s.dims_[rank_ - 2], s.dims_[rank_ - 1]);
    return s;
}

std::string Shape::to_string() const {
    std::string out = "(";
    for (size_t axis = 0; axis < rank_; axis++) {
        if (axis) out += ", ";
        out += std::to_string(dims_[axis]);
    }
    return out + ")";
}

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
    if (metal_mode()) {
        return Storage(metal::allocate(raw_bytes), Deleter{tensor_detail::Allocator::Metal});
    }
    if (raw_bytes >= kAlignThresholdBytes) {
        const size_t bytes = ((raw_bytes + kPageBytes - 1) / kPageBytes) * kPageBytes;
        void* p = nullptr;
        if (posix_memalign(&p, kPageBytes, bytes) != 0) {
            throw std::bad_alloc();
        }
        return Storage(static_cast<float*>(p), Deleter{tensor_detail::Allocator::PageAligned});
    }
    return Storage(new float[n], Deleter{tensor_detail::Allocator::Heap});
}

Tensor::Tensor(const Shape& shape) : Tensor(uninitialized(shape)) {
    const size_t bytes = numel() * sizeof(float);
    if (metal_mode() && bytes > kCpuZeroFillBytes) {
        metal::ops::fill(device_data(), numel(), 0.0f);
    } else {
        std::memset(data.get(), 0, bytes);
    }
}

Tensor::Tensor(size_t rows, size_t cols) : Tensor(Shape{rows, cols}) {}

Tensor::Tensor(size_t batch_size, size_t rows, size_t cols)
    : Tensor(Shape{batch_size, rows, cols}) {}

Tensor Tensor::uninitialized(const Shape& shape) {
    check_dims(shape);
    Tensor t;
    t.shape_ = shape;
    t.data = alloc_floats(shape.numel());
    return t;
}

Tensor Tensor::uninitialized(size_t rows, size_t cols) {
    return uninitialized(Shape{rows, cols});
}

Tensor Tensor::uninitialized(size_t batch_size, size_t rows, size_t cols) {
    return uninitialized(Shape{batch_size, rows, cols});
}

Tensor::Tensor(const Tensor& other) : shape_(other.shape_) {
    if (const size_t total = numel(); total > 0) {
        data = alloc_floats(total);
        // A tensor the GPU may still be writing is copied by the GPU, in
        // stream order, rather than by a CPU memcpy that would have to wait.
        if (other.device_visible_ && metal_mode()) {
            metal::ops::copy(other.device_data(), device_data(), total);
        } else {
            std::memcpy(data.get(), other.raw(), total * sizeof(float));
        }
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : data(std::move(other.data)),
      shape_(std::exchange(other.shape_, Shape{})),
      device_visible_(std::exchange(other.device_visible_, false)) {}

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
    swap(shape_, other.shape_);
    swap(device_visible_, other.device_visible_);
}

float Tensor::getValue(size_t row, size_t col) const {
    if (row >= getRows() || col >= getCols()) {
        throw std::out_of_range("Tensor(2D) index out of bounds");
    }
    fence();
    return data[row * getCols() + col];
}

void Tensor::setValue(size_t row, size_t col, float value) {
    if (row >= getRows() || col >= getCols()) {
        throw std::out_of_range("Tensor(2D) index out of bounds");
    }
    fence();
    data[row * getCols() + col] = value;
}

float Tensor::getValue(size_t batch, size_t row, size_t col) const {
    if (batch >= getBatchSize() || row >= getRows() || col >= getCols()) {
        throw std::out_of_range("Tensor(3D) index out of bounds");
    }
    fence();
    return data[(batch * getRows() + row) * getCols() + col];
}

void Tensor::setValue(size_t batch, size_t row, size_t col, float value) {
    if (batch >= getBatchSize() || row >= getRows() || col >= getCols()) {
        throw std::out_of_range("Tensor(3D) index out of bounds");
    }
    fence();
    data[(batch * getRows() + row) * getCols() + col] = value;
}

Tensor Tensor::matmul(const Tensor& other) const {
    assertValid("matmul(lhs)");
    other.assertValid("matmul(rhs)");

    const bool batched_rhs = other.getIs3D();
    const bool compatible =
        getCols() == other.getRows() && (!batched_rhs || same_leading_dims(shape_, other.shape_));
    if (rank() < 2 || other.rank() < 2 || !compatible) {
        throw std::invalid_argument("matmul: cannot multiply " + shape_.to_string() + " by "
                                    + other.shape_.to_string());
    }

    const size_t M = getRows();
    const size_t K = getCols();
    const size_t N = other.getCols();
    Tensor result = uninitialized(shape_.with_last_dim(N));

    // One sgemm per matrix, through the blas_sgemm_ex seam so the GPU
    // decision lives in one place. At current scale none reaches Metal:
    // the largest per-sequence product in the medium preset
    // (256x768 @ 768x3072) is 1.2 GFLOP against a 10 GFLOP threshold, so
    // each call runs the same cblas_sgemm it always did.
    const float* lhs = raw();
    const float* rhs = other.raw();
    float* out = result.raw();
    for (size_t b = 0; b < getBatchSize(); b++) {
        blas_sgemm_ex(lhs + b * M * K, rhs + (batched_rhs ? b * K * N : 0), out + b * M * N, M, N,
                      K, false, false, 1.0f, 0.0f);
    }
    return result;
}

Tensor Tensor::add(const Tensor& other) const {
    assertValid("add(lhs)");
    other.assertValid("add(rhs)");

    if (shape_ == other.shape_) {
        Tensor result = empty_like(*this);
        blas_vadd(raw(), other.raw(), result.raw(), numel());
        return result;
    }

    // A higher-rank operand fixes the result; two 2D operands may both
    // broadcast, to the larger extent on each axis.
    const bool fixed_lhs = getIs3D();
    const bool fixed_rhs = other.getIs3D();
    const Shape out_shape = fixed_lhs   ? shape_
                            : fixed_rhs ? other.shape_
                                        : Shape{std::max(getRows(), other.getRows()),
                                                std::max(getCols(), other.getCols())};
    const size_t R = out_shape[out_shape.rank() - 2];
    const size_t C = out_shape[out_shape.rank() - 1];

    auto fits = [&](const Tensor& t, bool fixed) {
        if (fixed) return t.shape_ == out_shape;
        return t.rank() == 2 && (t.getRows() == R || t.getRows() == 1)
               && (t.getCols() == C || t.getCols() == 1);
    };
    if (!fits(*this, fixed_lhs) || !fits(other, fixed_rhs)) {
        throw std::invalid_argument("add: shapes " + shape_.to_string() + " and "
                                    + other.shape_.to_string() + " do not broadcast");
    }

    Tensor result = uninitialized(out_shape);
    broadcast_add(broadcast_operand(*this, R, C), broadcast_operand(other, R, C), result.raw(),
                  result.getBatchSize(), R, C);
    return result;
}

Tensor Tensor::subtract(const Tensor& other) const {
    assertValid("subtract(lhs)");
    other.assertValid("subtract(rhs)");
    require_same_shape(*this, other, "subtract");
    Tensor result = empty_like(*this);
    blas_vsub(raw(), other.raw(), result.raw(), numel());
    return result;
}

Tensor Tensor::elementwise(const Tensor& other) const {
    assertValid("elementwise(lhs)");
    other.assertValid("elementwise(rhs)");
    require_same_shape(*this, other, "elementwise");
    Tensor result = empty_like(*this);
    blas_vmul(raw(), other.raw(), result.raw(), numel());
    return result;
}

Tensor Tensor::transpose() const {
    assertValid("transpose(this)");

    const size_t R = getRows();
    const size_t C = getCols();
    Tensor result = uninitialized(shape_.transposed());

    constexpr size_t kBlock = 32;
    for (size_t b = 0; b < getBatchSize(); b++) {
        const float* src = raw() + b * R * C;
        float* dst = result.raw() + b * C * R;
        for (size_t i0 = 0; i0 < R; i0 += kBlock) {
            const size_t i_max = std::min(i0 + kBlock, R);
            for (size_t j0 = 0; j0 < C; j0 += kBlock) {
                const size_t j_max = std::min(j0 + kBlock, C);
                for (size_t i = i0; i < i_max; ++i) {
                    for (size_t j = j0; j < j_max; ++j) {
                        dst[j * R + i] = src[i * C + j];
                    }
                }
            }
        }
    }
    return result;
}

Tensor Tensor::softmax() const {
    assertValid("softmax(this)");

    // Rows are contiguous at any rank, so one loop over the flat rows
    // covers every batch. exp goes through vec_exp (SIMD).
    Tensor result = empty_like(*this);
    const float* input_data = raw();
    float* output_data = result.raw();
    const size_t cols = getCols();

    parallel_for(getFlatRows(), 16, [&](size_t begin, size_t end) {
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
            vec_exp(row_out, row_out, cols);

            const float inv_sum = 1.0f / vec_sum(row_out, cols);
            for (size_t j = 0; j < cols; j++) {
                row_out[j] *= inv_sum;
            }
        }
    });
    return result;
}

void Tensor::fill(float value) {
    assertValid("fill(this)");
    blas_vfill(value, raw(), numel());
}

Tensor Tensor::scale(float scaler) const {
    assertValid("scale(this)");
    Tensor result = empty_like(*this);
    blas_vsmul(raw(), scaler, result.raw(), numel());
    return result;
}

Tensor Tensor::slice(size_t start_row, size_t num_rows, size_t start_col, size_t num_cols) const {
    assertValid("slice(this)");
    if (rank() != 2) throw std::invalid_argument("slice: 2D tensors only");

    if (start_row + num_rows > getRows() || start_col + num_cols > getCols()) {
        throw std::invalid_argument(
            "slice: rows [" + std::to_string(start_row) + ", "
            + std::to_string(start_row + num_rows) + ") x cols [" + std::to_string(start_col) + ", "
            + std::to_string(start_col + num_cols) + ") is outside " + shape_.to_string());
    }

    Tensor result = uninitialized(num_rows, num_cols);
    for (size_t i = 0; i < num_rows; i++) {
        std::memcpy(result.raw() + i * num_cols, raw() + (start_row + i) * getCols() + start_col,
                    num_cols * sizeof(float));
    }
    return result;
}

void Tensor::xavier(size_t fan_in, size_t fan_out) {
    assertValid("xavier(target)");

    float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    std::uniform_real_distribution<float> dis(-limit, limit);

    InitStream& stream = init_stream();
    std::lock_guard<std::mutex> lk(stream.mutex);
    for (float& v : values()) {
        v = dis(stream.gen);
    }
}

void Tensor::set_init_seed(uint64_t seed) {
    InitStream& stream = init_stream();
    std::lock_guard<std::mutex> lk(stream.mutex);
    // mt19937 takes a 32-bit seed; fold the high half in rather than drop it.
    stream.gen.seed(static_cast<std::mt19937::result_type>(seed ^ (seed >> 32)));
}

Tensor Tensor::create_causal_mask(size_t seq_len) {
    Tensor mask = uninitialized(seq_len, seq_len);
    float* m = mask.raw();
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < seq_len; j++) {
            m[i * seq_len + j] = j > i ? -1e9f : 0.0f;
        }
    }
    return mask;
}

void Tensor::assertValid(const std::string& context) const {
    // Storage exists exactly when the shape passed check_dims, so a null
    // pointer covers both an empty and a moved-from tensor.
    if (!data) {
        throw std::runtime_error("Tensor error [" + context + "]: no data (shape "
                                 + shape_.to_string() + ")");
    }
}

void Tensor::scale_inplace(float scalar) {
    assertValid("scale_inplace");
    float* p = raw();
    blas_vsmul(p, scalar, p, numel());
}

void Tensor::add_inplace(const Tensor& other) {
    assertValid("add_inplace");
    other.assertValid("add_inplace(other)");
    require_same_shape(*this, other, "add_inplace");
    float* p = raw();
    blas_vadd(p, other.raw(), p, numel());
}

void Tensor::multiply_inplace(const Tensor& other) {
    assertValid("multiply_inplace");
    other.assertValid("multiply_inplace(other)");
    require_same_shape(*this, other, "multiply_inplace");
    float* p = raw();
    blas_vmul(p, other.raw(), p, numel());
}

void Tensor::zero() {
    if (numel() > 0) std::memset(raw(), 0, numel() * sizeof(float));
}

}  // namespace grad
