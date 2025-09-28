#include "transformer/tensor.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

Tensor::Tensor() {
    this->rows = 1;
    this->cols = 1; 
    this->batch_size = 1;
    this->is_3d = false;
    this->data = new float[1];
    this->data[0] = 0.0f;
}

Tensor::Tensor(int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    this->batch_size = 1;
    this->is_3d = false;
    this->data = new float[batch_size * rows * cols];

    for (int i = 0; i < batch_size * rows * cols; i++) {
        data[i] = 0.0f;
    }
}

Tensor::Tensor(int batch_size, int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    this->batch_size = batch_size;
    this->is_3d = true;
    this->data = new float[batch_size * rows * cols];

    for (int i = 0; i < batch_size * rows * cols; i++) {
        data[i] = 0.0f;
    }
}

Tensor::Tensor(const Tensor& other) {
    this->rows = other.rows;
    this->cols = other.cols;
    this->batch_size = other.batch_size;
    this->is_3d = other.is_3d;
    this->data = new float[batch_size * rows * cols];

    for (int i = 0; i < batch_size * rows * cols; i++) {
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
    this->batch_size = other.batch_size;
    this->is_3d = other.is_3d;
    this->data = new float[batch_size * rows * cols];

    for (int i = 0; i < batch_size * rows * cols; i++) {
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

float Tensor::getValue(int batch, int row, int col) const {
    if (batch >= batch_size || row >= rows || col >= cols) {
        throw std::out_of_range("Tensor index out of bounds");
    }
    return data[batch * rows * cols + row * cols + col];
}

void Tensor::setValue(int row, int col, float value) {
    data[row * this->cols + col] = value;
}

void Tensor::setValue(int batch, int row, int col, float value) {
    if (batch >= batch_size || row >= rows || col >= cols) {
        throw std::out_of_range("Tensor index out of bounds");
    }
    data[batch * rows * cols + row * cols + col] = value;
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
    if (!this->is_3d && !other.is_3d) {
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
    } else if (this->is_3d && !other.is_3d) {
        if (this->cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions do not match for batch multiplication");
        }
        
        Tensor result(this->batch_size, this->rows, other.cols);
        for (int b = 0; b < this->batch_size; b++) {
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < other.cols; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < this->cols; k++) {
                        sum += this->getValue(b, i, k) * other.getValue(k, j);
                    }
                    result.setValue(b, i, j, sum);
                }
            }
        }
        return result;
    } else if (this->is_3d && other.is_3d) {
        if (this->batch_size != other.batch_size || this->cols != other.rows) {
            throw std::invalid_argument("Batch matrix dimensions do not match");
        }
        
        Tensor result(this->batch_size, this->rows, other.cols);
        for (int b = 0; b < this->batch_size; b++) {
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < other.cols; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < this->cols; k++) {
                        sum += this->getValue(b, i, k) * other.getValue(b, k, j);
                    }
                    result.setValue(b, i, j, sum);
                }
            }
        }
        return result;
    } else {
        throw std::invalid_argument("Unsupported matrix multiplication configuration");
    }
}

Tensor Tensor::add(const Tensor& other) const {
    if (!this->is_3d && !other.is_3d) {
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
    } else if (this->is_3d && !other.is_3d) {
        if (this->rows != other.rows || this->cols != other.cols) {
            throw std::invalid_argument("Tensor dimensions don't match for broadcasting");
        }
        
        Tensor result(this->batch_size, this->rows, this->cols);
        for (int b = 0; b < this->batch_size; b++) {
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    result.setValue(b, i, j, this->getValue(b, i, j) + other.getValue(i, j));
                }
            }
        }
        return result;
    } else if (!this->is_3d && other.is_3d) {
        if (this->rows != other.rows || this->cols != other.cols) {
            throw std::invalid_argument("Tensor dimensions don't match for broadcasting");
        }
        
        Tensor result(other.batch_size, this->rows, this->cols);
        for (int b = 0; b < other.batch_size; b++) {
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    result.setValue(b, i, j, this->getValue(i, j) + other.getValue(b, i, j));
                }
            }
        }
        return result;
    } else if (this->is_3d && other.is_3d) {
        if (this->batch_size != other.batch_size || this->rows != other.rows || this->cols != other.cols) {
            throw std::invalid_argument("3D tensor dimensions don't match");
        }
        
        Tensor result(this->batch_size, this->rows, this->cols);
        for (int b = 0; b < this->batch_size; b++) {
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    result.setValue(b, i, j, this->getValue(b, i, j) + other.getValue(b, i, j));
                }
            }
        }
        return result;
    } else {
        throw std::invalid_argument("Unsupported addition configuration");
    }
}

Tensor Tensor::subtract(const Tensor& other) const {
    if (!this->is_3d && !other.is_3d) {
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
    } else if (this->is_3d && other.is_3d) {
        if (this->batch_size != other.batch_size || this->rows != other.rows || this->cols != other.cols) {
            throw std::invalid_argument("3D tensor dimensions don't match for subtraction");
        }
        Tensor result(this->batch_size, this->rows, this->cols);
        for (int b = 0; b < this->batch_size; b++) {
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    result.setValue(b, i, j, this->getValue(b, i, j) - other.getValue(b, i, j));
                }
            }
        }
        return result;
    } else {
        throw std::invalid_argument("Cannot subtract tensors with different dimensionalities");
    }
}

Tensor Tensor::elementwise(const Tensor& other) const {
    if (!this->is_3d && !other.is_3d) {
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
    } else if (this->is_3d && other.is_3d) {
        if (this->batch_size != other.batch_size || this->rows != other.rows || this->cols != other.cols) {
            throw std::invalid_argument("3D tensor dimensions don't match for elementwise multiply");
        }
        Tensor result(this->batch_size, this->rows, this->cols);
        for (int b = 0; b < this->batch_size; b++) {
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    result.setValue(b, i, j, this->getValue(b, i, j) * other.getValue(b, i, j));
                }
            }
        }
        return result;
    } else {
        throw std::invalid_argument("Cannot perform elementwise multiply on tensors with different dimensionalities");
    }
}

Tensor Tensor::transpose() const {
    if (!this->is_3d) {
        Tensor result(this->cols, this->rows);
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                result.setValue(j, i, this->getValue(i, j));
            }
        }
        return result;
    } else {
        Tensor result(this->batch_size, this->cols, this->rows);
        for (int b = 0; b < this->batch_size; b++) {
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    result.setValue(b, j, i, this->getValue(b, i, j));
                }
            }
        }
        return result;
    }
}

Tensor Tensor::softmax() const {
    if (!this->is_3d) {
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
    } else {
        Tensor result(this->batch_size, this->rows, this->cols);
        for (int b = 0; b < this->batch_size; b++) {
            for (int i = 0; i < this->rows; i++) {
                float max_val = getValue(b, i, 0);
                for (int j = 1; j < this->cols; j++) {
                    max_val = std::max(max_val, getValue(b, i, j));
                }
                float sum = 0.0f;
                for (int j = 0; j < this->cols; j++) {
                    sum += exp(getValue(b, i, j) - max_val);
                }
                for (int j = 0; j < this->cols; j++) {
                    result.setValue(b, i, j, exp(getValue(b, i, j) - max_val) / sum);
                }
            }
        }
        return result;
    }
}

void Tensor::fill(float value) {
    for (int i = 0; i < this->batch_size * this->rows * this->cols; i++) {
       data[i] = value;
    }
}

Tensor Tensor::scale(float scaler) const {
    if (!this->is_3d) {
        Tensor result(this->rows, this->cols);
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                result.setValue(i, j, this->getValue(i, j) * scaler);
            }
        }
        return result;
    } else {
        Tensor result(this->batch_size, this->rows, this->cols);
        for (int b = 0; b < this->batch_size; b++) {
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    result.setValue(b, i, j, this->getValue(b, i, j) * scaler);
                }
            }
        }
        return result;
    }
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

    for (int i = 0; i < batch_size * rows * cols; i++) {
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

Tensor Tensor::create_casual_mask(int seq_len) {
    Tensor mask(seq_len, seq_len);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            if (j > i) {
                mask.setValue(i, j, -1e9f);
            } else {
                mask.setValue(i, j, 0.0f);
            }
        }
    }
    return mask;
}

Tensor Tensor::create_casual_mask_batch(int batch_size, int seq_len) {
    Tensor mask(batch_size, seq_len, seq_len);
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                if (j > i) {
                    mask.setValue(b, i, j, -1e9f);
                } else {
                    mask.setValue(b, i, j, 0.0f);
                }
            }
        }
    }
    return mask;
}