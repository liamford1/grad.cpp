#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

constexpr size_t MAX_TENSOR_ELEMENTS = 1ULL << 30;

namespace tensor_detail {
// Releases a Tensor buffer with whichever allocator produced it. Two are in
// play (page-aligned for large tensors so Metal can wrap them zero-copy,
// plain new[] below that - see Tensor::alloc_floats), and stamping the
// choice into the deleter at allocation time means the release path can
// never disagree with it.
//
// This lives at namespace scope deliberately: libc++ constrains
// unique_ptr's default constructor on is_default_constructible<Deleter>,
// and a private nested deleter fails that access check while the enclosing
// class is being completed, which silently deletes Tensor's own defaulted
// default constructor (-Wdefaulted-function-deleted).
struct PageAwareDeleter {
    bool page_aligned = false;
    void operator()(float* p) const noexcept {
        if (page_aligned) std::free(p); else delete[] p;
    }
};
}  // namespace tensor_detail

// Dense row-major float storage, 2D (rows x cols) or 3D (batch x rows x cols),
// owned through a unique_ptr carrying a PageAwareDeleter.
class Tensor {
    private:
        using Deleter = tensor_detail::PageAwareDeleter;
        using Storage = std::unique_ptr<float[], Deleter>;

        Storage data;
        size_t rows = 0, cols = 0, batch_size = 0;
        bool is_3d = false;

        [[nodiscard]] static Storage alloc_floats(size_t n);
    public:
        Tensor() = default;
        Tensor(size_t rows, size_t cols);
        Tensor(size_t batch_size, size_t rows, size_t cols);

        // Allocation without the zero-fill, for outputs whose every element
        // is written before being read (beta=0 matmuls, elementwise
        // producers). The plain constructors zero because accumulation
        // targets (gradients, Adam moments) rely on it; that memset was
        // ~6% of training-step CPU when applied to everything
        // (BENCHMARKS.md #10).
        [[nodiscard]] static Tensor uninitialized(size_t rows, size_t cols);
        [[nodiscard]] static Tensor uninitialized(size_t batch_size, size_t rows, size_t cols);

        // Copies are deep. Moves are O(1) and leave the source empty
        // (numel() == 0), which the autograd graph reads as "released".
        Tensor(const Tensor& other);
        Tensor& operator=(const Tensor& other);
        Tensor(Tensor&& other) noexcept;
        Tensor& operator=(Tensor&& other) noexcept;
        ~Tensor() = default;

        void swap(Tensor& other) noexcept;
        friend void swap(Tensor& a, Tensor& b) noexcept { a.swap(b); }

        [[nodiscard]] float getValue(size_t row, size_t col) const;
        [[nodiscard]] float getValue(size_t batch, size_t row, size_t col) const;
        void setValue(size_t row, size_t col, float value);
        void setValue(size_t batch, size_t row, size_t col, float value);

        [[nodiscard]] Tensor matmul(const Tensor& other) const;
        [[nodiscard]] Tensor add(const Tensor& other) const;
        [[nodiscard]] Tensor subtract(const Tensor& other) const;
        [[nodiscard]] Tensor elementwise(const Tensor& other) const;

        void scale_inplace(float scalar);
        void add_inplace(const Tensor& other);
        void multiply_inplace(const Tensor& other);
        void zero();

        [[nodiscard]] Tensor transpose() const;
        [[nodiscard]] Tensor softmax() const;
        void fill(float value);
        [[nodiscard]] Tensor scale(float scaler) const;

        [[nodiscard]] Tensor reshape(size_t new_rows, size_t new_cols) const;
        [[nodiscard]] Tensor slice(size_t start_row, size_t num_rows, size_t start_col, size_t num_cols) const;

        // Fills with U(-limit, limit), limit = sqrt(6 / (fan_in + fan_out)),
        // drawn from one process-wide mt19937. The generator starts from a
        // fixed seed, so model construction is reproducible; set_init_seed
        // reseeds it (call before building the model).
        void xavier(size_t fan_in, size_t fan_out);
        static void set_init_seed(uint64_t seed);
        [[nodiscard]] static Tensor create_causal_mask(size_t seq_len);
        [[nodiscard]] static Tensor create_causal_mask_batch(size_t batch_size, size_t seq_len);

        [[nodiscard]] size_t getRows() const noexcept { return rows; }
        [[nodiscard]] size_t getCols() const noexcept { return cols; }
        [[nodiscard]] size_t getBatchSize() const noexcept { return batch_size; }
        [[nodiscard]] bool getIs3D() const noexcept { return is_3d; }
        [[nodiscard]] size_t numel() const noexcept { return is_3d ? batch_size * rows * cols : rows * cols; }

        void assertValid(const std::string& context = "") const;

        [[nodiscard]] float* raw() noexcept { return data.get(); }
        [[nodiscard]] const float* raw() const noexcept { return data.get(); }
};
