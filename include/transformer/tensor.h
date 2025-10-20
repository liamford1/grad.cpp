#pragma once
#include <string>
#include <cstddef>

constexpr size_t MAX_TENSOR_ELEMENTS = 1ULL << 30;

class Tensor {
    private:
        float* data;
        size_t rows, cols, batch_size;
        bool is_3d;
    public:
        Tensor();
        Tensor(size_t rows, size_t cols);
        Tensor(size_t batch_size, size_t rows, size_t cols);
        Tensor(const Tensor& other);
        Tensor& operator=(const Tensor& other);
        Tensor(Tensor&& other) noexcept;
        Tensor& operator=(Tensor&& other) noexcept;
        ~Tensor();

        float getValue(size_t row, size_t col) const;
        float getValue(size_t batch, size_t row, size_t col) const;
        void setValue(size_t row, size_t col, float value);
        void setValue(size_t batch, size_t row, size_t col, float value);
        void display() const;

        Tensor matmul(const Tensor& other) const;
        Tensor add(const Tensor& other) const;
        Tensor subtract(const Tensor& other) const;
        Tensor elementwise(const Tensor& other) const;

        void scale_inplace(float scalar);
        void add_inplace(const Tensor& other);
        void multiply_inplace(const Tensor& other);
        void zero();

        Tensor transpose() const;
        Tensor softmax() const;
        void fill(float value);
        Tensor scale(float scaler) const;

        Tensor reshape(size_t new_rows, size_t new_cols) const;
        Tensor slice(size_t start_row, size_t num_rows, size_t start_col, size_t num_cols) const;
        Tensor concatenate(const Tensor& other, int axis) const;

        void xavier(size_t fan_in, size_t fan_out);
        static Tensor create_causal_mask(size_t seq_len);
        static Tensor create_causal_mask_batch(size_t batch_size, size_t seq_len);

        size_t getRows() const { return rows; }
        size_t getCols() const { return cols; }
        size_t getBatchSize() const { return batch_size; }
        bool getIs3D() const { return is_3d; }
        size_t numel() const { return is_3d ? batch_size * rows * cols : rows * cols;}

        void assertValid(const std::string& context = "") const;

        float* raw() { return data; }
        const float* raw() const { return data; }
};
