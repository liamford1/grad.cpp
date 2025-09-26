#ifndef TENSOR_H
#define TENSOR_H

class Tensor {
    private:
        float* data;
        int rows;
        int cols;
    public:
        Tensor(int rows, int cols);
        Tensor(const Tensor& other);
        Tensor& operator=(const Tensor& other);
        ~Tensor();

        float getValue(int row, int col) const;
        void setValue(int row, int col, float value);
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
        Tensor causal_mask() const;
        
        int getRows() const { return rows; }
        int getCols() const { return cols; }
};

#endif
