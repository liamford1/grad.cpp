#pragma once
#include <string>

class Tensor {
    private:
        float* data;
        int rows, cols, batch_size;
        bool is_3d;
    public:
        Tensor();
        Tensor(int rows, int cols);
        Tensor(int batch_size, int rows, int cols);
        Tensor(const Tensor& other);
        Tensor& operator=(const Tensor& other);
        Tensor(Tensor&& other) noexcept;
        Tensor& operator=(Tensor&& other) noexcept;
        ~Tensor();

        float getValue(int row, int col) const;
        float getValue(int batch, int row, int col) const;
        void setValue(int row, int col, float value);
        void setValue(int batch, int row, int col, float value);
        void display() const;

        Tensor matmul(const Tensor& other) const;
        Tensor add(const Tensor& other) const;
        Tensor subtract(const Tensor& other) const;
        Tensor elementwise(const Tensor& other) const;

        Tensor transpose() const;
        Tensor softmax() const;
        void fill(float value);
        Tensor scale(float scaler) const;

        Tensor reshape(int new_rows, int new_cols) const;
        Tensor slice(int start_row, int num_rows, int start_col, int num_cols) const;
        Tensor concatenate(const Tensor& other, int axis) const;

        void xavier(int fan_in, int fan_out);
        static Tensor create_causal_mask(int seq_len);
        static Tensor create_causal_mask_batch(int batch_size, int seq_len);
        
        int getRows() const { return rows; }
        int getCols() const { return cols; }
        int getBatchSize() const { return batch_size; }
        bool getIs3D() const { return is_3d; }
        int numel() const { return is_3d ? batch_size * rows * cols : rows * cols;}

        void assertValid(const std::string& context = "") const;

        float* raw() { return data; }
        const float* raw() const { return data; }
};
