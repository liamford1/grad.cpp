#ifndef TENSOR_H
#define TENSOR_H

class Tensor {
    private:
        float* data;
        int rows;
        int cols;
    public:
        Tensor(int rows, int cols);

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
        Tensor relu() const;
        void fill(float value);
        Tensor scale(float scaler) const;
        
        int getRows() const { return rows; }
        int getCols() const { return cols; }
};

#endif
