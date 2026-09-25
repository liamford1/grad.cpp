#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <memory>
#include <span>
#include <string>

namespace grad {

// Size convention, for all of grad::core: extents, element counts, offsets
// and indices into memory are size_t, which is all Shape and Tensor use.
// Model and training hyperparameters (vocab_size, d_model, num_heads,
// max_len, seq_length, batch_size, step counts) are int: they come from
// CLI flags, presets and the checkpoint's int32 header. Each module checks
// its hyperparameters positive where they enter and converts them to
// size_t once; kernels take their extents from tensor shapes. Token ids
// are int. Narrowing back to int happens only where an interface demands
// it: the BLAS calls in blas_wrapper.h and the on-disk formats, through
// grad::narrow (grad/utils/narrow.h), which throws instead of wrapping.

// Largest element count a Tensor allocates (4 GiB of floats); a larger
// shape throws std::overflow_error at construction.
inline constexpr size_t kMaxTensorElements = size_t{1} << 30;
// The BLAS wrapper relies on this to pass any tensor length as an int.
static_assert(kMaxTensorElements <= static_cast<size_t>(std::numeric_limits<int>::max()));

// The extent of a dense, contiguous, row-major tensor: up to kMaxRank
// dimensions, outermost first. Strides are implicit. Rank 0 is the empty
// shape of a default-constructed or moved-from Tensor (numel 0), not a
// scalar; scalars here are (1, 1). Dimensions past rank() are held at 0,
// so equality is plain member-wise comparison.
class Shape {
    public:
        static constexpr size_t kMaxRank = 4;

        constexpr Shape() noexcept = default;
        // Throws std::invalid_argument for more than kMaxRank dimensions.
        // Zero or oversized extents are accepted here and rejected by the
        // Tensor that allocates the shape.
        Shape(std::initializer_list<size_t> dims);

        [[nodiscard]] constexpr size_t rank() const noexcept { return rank_; }
        [[nodiscard]] constexpr size_t operator[](size_t axis) const noexcept { return dims_[axis]; }
        [[nodiscard]] constexpr size_t numel() const noexcept { return numel_; }

        // The same shape with its innermost dimension replaced: the output
        // shape of a row-wise map to a different width (a projection, the
        // logits). Requires rank() > 0.
        [[nodiscard]] Shape with_last_dim(size_t n) const noexcept;
        // The same shape with its two innermost dimensions swapped (the
        // shape of a batched matrix transpose). Rank 0 and 1 are returned
        // unchanged.
        [[nodiscard]] Shape transposed() const noexcept;

        // "(2, 3, 4)", or "()" for the empty shape; for error messages.
        [[nodiscard]] std::string to_string() const;

        friend constexpr bool operator==(const Shape&, const Shape&) noexcept = default;

    private:
        std::array<size_t, kMaxRank> dims_{};
        size_t rank_ = 0;
        // Cached product of the dimensions. It may wrap for absurd extents,
        // which Tensor's allocation check catches from the dimensions.
        size_t numel_ = 0;
};

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

// Dense row-major float storage of any Shape up to Shape::kMaxRank,
// owned through a unique_ptr carrying a PageAwareDeleter. The model code
// works in 2D (rows, cols) and 3D (batch, rows, cols) terms, which the
// getRows/getCols/getBatchSize/getIs3D accessors present as a view over
// the shape; elementwise ops and reductions go through numel() and never
// branch on rank.
class Tensor {
    private:
        using Deleter = tensor_detail::PageAwareDeleter;
        using Storage = std::unique_ptr<float[], Deleter>;

        Storage data;
        Shape shape_;

        [[nodiscard]] static Storage alloc_floats(size_t n);
    public:
        Tensor() = default;
        explicit Tensor(const Shape& shape);
        Tensor(size_t rows, size_t cols);
        Tensor(size_t batch_size, size_t rows, size_t cols);

        // Allocation without the zero-fill, for outputs whose every element
        // is written before being read (beta=0 matmuls, elementwise
        // producers). The plain constructors zero because accumulation
        // targets (gradients, Adam moments) rely on it; that memset was
        // ~6% of training-step CPU when applied to everything
        // (BENCHMARKS.md #10).
        [[nodiscard]] static Tensor uninitialized(const Shape& shape);
        [[nodiscard]] static Tensor uninitialized(size_t rows, size_t cols);
        [[nodiscard]] static Tensor uninitialized(size_t batch_size, size_t rows, size_t cols);

        // Same shape as t: empty_like leaves the values uninitialized,
        // zeros_like zero-fills.
        [[nodiscard]] static Tensor empty_like(const Tensor& t) { return uninitialized(t.shape_); }
        [[nodiscard]] static Tensor zeros_like(const Tensor& t) { return Tensor(t.shape_); }

        // Copies are deep. Moves are O(1) and leave the source empty
        // (numel() == 0), which the autograd graph reads as "released".
        Tensor(const Tensor& other);
        Tensor& operator=(const Tensor& other);
        Tensor(Tensor&& other) noexcept;
        Tensor& operator=(Tensor&& other) noexcept;
        ~Tensor() = default;

        void swap(Tensor& other) noexcept;
        friend void swap(Tensor& a, Tensor& b) noexcept { a.swap(b); }

        // Bounds-checked element access through the 2D/3D view below; for
        // tests and cold paths, never inner loops.
        [[nodiscard]] float getValue(size_t row, size_t col) const;
        [[nodiscard]] float getValue(size_t batch, size_t row, size_t col) const;
        void setValue(size_t row, size_t col, float value);
        void setValue(size_t batch, size_t row, size_t col, float value);

        // (..., M, K) @ (K, N) -> (..., M, N), the right operand shared by
        // every leading index, or batched (..., M, K) @ (..., K, N) with
        // equal leading dimensions.
        [[nodiscard]] Tensor matmul(const Tensor& other) const;
        // Only a 2D operand broadcasts (a size-1 row or column, and across
        // every leading index of the other operand); a higher-rank operand
        // fixes the result shape.
        [[nodiscard]] Tensor add(const Tensor& other) const;
        // subtract and elementwise (product) require equal shapes.
        [[nodiscard]] Tensor subtract(const Tensor& other) const;
        [[nodiscard]] Tensor elementwise(const Tensor& other) const;

        void scale_inplace(float scalar);
        void add_inplace(const Tensor& other);
        void multiply_inplace(const Tensor& other);
        void zero();

        // Swaps the two innermost dimensions of every matrix.
        [[nodiscard]] Tensor transpose() const;
        // Softmax over the innermost dimension.
        [[nodiscard]] Tensor softmax() const;
        void fill(float value);
        [[nodiscard]] Tensor scale(float scaler) const;

        [[nodiscard]] Tensor slice(size_t start_row, size_t num_rows, size_t start_col, size_t num_cols) const;

        // Fills with U(-limit, limit), limit = sqrt(6 / (fan_in + fan_out)),
        // drawn from one process-wide mt19937. The generator starts from a
        // fixed seed, so model construction is reproducible; set_init_seed
        // reseeds it (call before building the model).
        void xavier(size_t fan_in, size_t fan_out);
        static void set_init_seed(uint64_t seed);
        [[nodiscard]] static Tensor create_causal_mask(size_t seq_len);

        [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
        [[nodiscard]] size_t rank() const noexcept { return shape_.rank(); }
        [[nodiscard]] size_t numel() const noexcept { return shape_.numel(); }

        // The 2D/3D view used by the model code. cols is the innermost
        // dimension and rows the one outside it; every dimension further
        // out folds into the batch, so a rank-2 tensor has batch 1 and is
        // not 3D, and any higher rank reads as (batch, rows, cols). An
        // empty tensor reports 0 for all three.
        [[nodiscard]] size_t getCols() const noexcept {
            return rank() > 0 ? shape_[rank() - 1] : 0;
        }
        [[nodiscard]] size_t getRows() const noexcept {
            return rank() > 1 ? shape_[rank() - 2] : (rank() == 1 ? 1 : 0);
        }
        [[nodiscard]] size_t getBatchSize() const noexcept {
            if (rank() == 0) return 0;
            size_t batch = 1;
            for (size_t axis = 0; axis + 2 < rank(); axis++) batch *= shape_[axis];
            return batch;
        }
        [[nodiscard]] bool getIs3D() const noexcept { return rank() > 2; }
        // Rows of the (numel / cols, cols) matrix the tensor is in memory:
        // batch * rows. Row-wise ops (softmax, norms, projections) run
        // over these without caring about the batch structure.
        [[nodiscard]] size_t getFlatRows() const noexcept {
            return rank() > 0 ? numel() / getCols() : 0;
        }

        void assertValid(const std::string& context = "") const;

        [[nodiscard]] float* raw() noexcept { return data.get(); }
        [[nodiscard]] const float* raw() const noexcept { return data.get(); }
        [[nodiscard]] std::span<float> values() noexcept { return {data.get(), numel()}; }
        [[nodiscard]] std::span<const float> values() const noexcept { return {data.get(), numel()}; }
};

}  // namespace grad
