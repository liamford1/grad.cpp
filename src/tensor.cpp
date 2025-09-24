#include "tensor.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>

Tensor::Tensor(int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    this->data = new float[rows * cols];

    for (int i = 0; i < rows * cols; i++) {
        data[i] = 0.0f;
    }
}

Tensor::Tensor(const Tensor& other) {
    this->rows = other.rows;
    this->cols = other.cols;
    this->data = new float[rows * cols];

    for (int i = 0; i < rows * cols; i++) {
        this->data[i] = other.data[i];
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) {
        return *this;
    }

    delete[] data;

    this->rows = other.rows;
    this->cols = other.cols;
    this->data = new float[rows * cols];

    for (int i = 0; i < rows * cols; i++) {
        this->data[i] = other.data[i];
    }

    return *this;
}

Tensor::~Tensor() {
    delete[] data;
}

float Tensor::getValue(int row, int col) const {
    return data[row * this->cols + col];
}

void Tensor::setValue(int row, int col, float value) {
    data[row * this->cols + col] = value;
}

void Tensor::display() const {
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            std::cout << getValue(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (this->cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }

    Tensor result(this->rows, other.cols);

    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < this->cols; k++) {
                sum += this->getValue(i, k) * other.getValue(k, j);
            }
            result.setValue(i, j, sum);
        }
    }

    return result;
}

Tensor Tensor::add(const Tensor& other) const {
    bool rows_compatible = (rows == other.rows) || (rows == 1) || (other.rows == 1);
    bool cols_compatible = (cols == other.cols) || (cols == 1) || (other.cols == 1);

    if (!rows_compatible || !cols_compatible) {
        throw std::invalid_argument("Shapes not broadcastable");
    }

    int result_rows = std::max(this->rows, other.rows);
    int result_cols = std::max(this->cols, other.cols);
    Tensor result(result_rows, result_cols);

    for (int i = 0; i < result_rows; i++) {
        for (int j = 0; j < result_cols; j++) {
            int this_i = (this->rows == 1) ? 0 : i;
            int this_j = (this->cols == 1) ? 0 : j;
            int other_i = (other.rows == 1) ? 0 : i;
            int other_j = (other.cols == 1) ? 0 : j;
            
            result.setValue(i, j, this->getValue(this_i, this_j) + other.getValue(other_i, other_j));
        }
    }

    return result;
}

Tensor Tensor::subtract(const Tensor& other) const {
    if (this->rows != other.rows || this->cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for subtraction");
    }

    Tensor result(this->rows, this->cols);

    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            result.setValue(i, j, this->getValue(i, j) - other.getValue(i, j));
        }
    }

    return result;
}

Tensor Tensor::elementwise(const Tensor& other) const {
    if (this->rows != other.rows || this->cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for elementwise multiply");
    }

    Tensor result(this->rows, this->cols);

    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            result.setValue(i, j, this->getValue(i, j) * other.getValue(i, j));
        }
    }

    return result;
}

Tensor Tensor::transpose() const {
    Tensor result(this->cols, this->rows);

    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            result.setValue(j, i, this->getValue(i, j));
        }
    }

    return result;
}

Tensor Tensor::relu() const {
    Tensor result(this->rows, this->cols);

    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            result.setValue(i, j, std::max(0.0f, this->getValue(i, j)));
        }
    }

    return result;
}

Tensor Tensor::softmax() const {
    Tensor result(this->rows, this->cols);
    
    for (int i = 0; i < this->rows; i++) {

        float max_val = getValue(i, 0);
        for (int j = 0; j < this->cols; j++) {
            max_val = std::max(max_val, getValue(i, j));
        }

        float sum = 0.0f;
        for (int j = 0; j < this->cols; j++) {
            sum += exp(getValue(i, j) - max_val);
        }

        for (int j = 0; j < this->cols; j++) {
            result.setValue(i, j, exp(getValue(i, j) - max_val) / sum);
        }
    }

    return result;
}

void Tensor::fill(float value) {
    for (int i = 0; i < this->rows * this->cols; i++) {
       data[i] = value;
    }
}

Tensor Tensor::scale(float scaler) const {
    Tensor result(this->rows, this->cols);

    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            result.setValue(i, j, this->getValue(i ,j) * scaler);
        }
    }

    return result;
}

Tensor Tensor::reshape(int new_rows, int new_cols) const {
    if (new_rows * new_cols != this->rows * this->cols) {
        throw std::invalid_argument("Matrix sizes do not match for reshape");
    }

    Tensor result(new_rows, new_cols);

    for (int i = 0; i < new_rows * new_cols; i++) {
            result.setValue(i / new_cols, i % new_cols, data[i]);
    }
    return result;
}

Tensor Tensor::slice(int start_row, int num_rows, int start_col, int num_cols) const {
    if (start_row + num_rows > this->rows || start_col + num_cols > this->cols) {
        throw std::invalid_argument("Out of bounds error");
    }
    if (start_row < 0 || num_rows < 0 || start_col < 0 || num_cols < 0) {
        throw std::invalid_argument("Error there are negative parameters");
    }

    Tensor result(num_rows, num_cols);

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            result.setValue(i, j, this->getValue(start_row + i, start_col + j));
        }
    }

    return result;
}

Tensor Tensor::concatenate(const Tensor& other, int axis) const {
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
        
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                result.setValue(i, j, this->getValue(i, j));
            }
        }
        
        for (int i = 0; i < other.rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                result.setValue(this->rows + i, j, other.getValue(i, j));
            }
        }
        
        return result;
    } else {
        Tensor result(this->rows, this->cols + other.cols);
        
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                result.setValue(i, j, this->getValue(i, j));
            }
        }
        
        for (int i = 0; i < other.rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                result.setValue(i, this->cols + j, other.getValue(i, j));
            }
        }
        
        return result;
    }
}

void Tensor::xavier(int fan_in, int fan_out) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    float limit = std::sqrt(6.0f / (fan_in + fan_out));
    std::uniform_real_distribution<float> dis(-limit, limit);

    for (int i = 0; i < rows * cols; i++) {
        data[i] = dis(gen);
    }
}

Tensor Tensor::causal_mask() const {
    Tensor result(this->rows, this->cols);

    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            if (j <= i) {
                result.setValue(i, j, getValue(i, j));
            } else {
                result.setValue(i, j, -1e9f);
            }
        }
    }
    return result;
}