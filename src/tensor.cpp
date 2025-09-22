#include "tensor.h"
#include <iostream>
#include <algorithm>
#include <cmath>

Tensor::Tensor(int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    this->data = new float[rows * cols];

    for (int i = 0; i < rows * cols; i++) {
        data[i] = 0.0f;
    }
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
    if (this->rows != other.rows || this->cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for addition");
    }

    Tensor result(this->rows, this->cols);

    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            result.setValue(i, j, this->getValue(i ,j) + other.getValue(i ,j));
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
        float sum = 0.0f;

        for (int j = 0; j < this->cols; j++) {
            sum += exp(getValue(i, j));
        }

        for (int j = 0; j < this->cols; j++) {
            result.setValue(i, j, exp(getValue(i, j)) / sum);
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