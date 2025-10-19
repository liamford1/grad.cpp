#include "transformer/tensor.h"
#include "transformer/blas_wrapper.h"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>
#include <cassert>
#include <string>

Tensor::Tensor() {
    this->rows = 0;
    this->cols = 0; 
    this->batch_size = 0;
    this->is_3d = false;
    this->data = nullptr;
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

Tensor::Tensor(Tensor&& other) noexcept :
    data(other.data),
    rows(other.rows),
    cols(other.cols),
    batch_size(other.batch_size),
    is_3d(other.is_3d) 
{
    other.data = nullptr;
    other.rows = other.cols = other.batch_size = 0;
    other.is_3d = false;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;
    Tensor tmp(other);
    
    std::swap(data, tmp.data);
    std::swap(rows, tmp.rows);
    std::swap(cols, tmp.cols);
    std::swap(batch_size, tmp.batch_size);
    std::swap(is_3d, tmp.is_3d);
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;
    delete[] data;
    data = other.data;
    rows = other.rows;
    cols = other.cols;
    batch_size = other.batch_size;
    is_3d = other.is_3d;
    other.data = nullptr;
    other.rows = other.cols = other.batch_size = 0;
    other.is_3d = false;
    return *this;
}

Tensor::~Tensor() {
    delete[] data;
}

//2D Tensor methods
float Tensor::getValue(int row, int col) const {
    if (row < 0 || col < 0 || row >= rows || col >= cols) {
        throw std::out_of_range("Tensor(2D) index out of bounds");
    }
    return data[row * cols + col];
}

void Tensor::setValue(int row, int col, float value) {
    if (row < 0 || col < 0 || row >= rows || col >= cols) {
        throw std::out_of_range("Tensor(2D) index out of bounds");
    }
    data[row * cols + col] = value;
}

//3D Tensor methods
float Tensor::getValue(int batch, int row, int col) const {
    if (batch < 0 || row < 0 || col < 0 ||
        batch >= batch_size || row >= rows || col >= cols) {
        throw std::out_of_range("Tensor(3D) index out of bounds");
    }
    return data[batch * rows * cols + row * cols + col];
}

void Tensor::setValue(int batch, int row, int col, float value) {
    if (batch < 0 || row < 0 || col < 0 ||
        batch >= batch_size || row >= rows || col >= cols) {
        throw std::out_of_range("Tensor(3D) index out of bounds");
    }
    data[batch * rows * cols + row * cols + col] = value;
}

void Tensor::display() const {
    assertValid("display(this)");
    if (is_3d) {
        std::cout << "[display] showing batch 0 of " << batch_size << "\n";
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << getValue(0, i, j) << " ";
            }
            std::cout << "\n";
        }
        return;
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) std::cout << getValue(i, j) << " ";
        std::cout << "\n";
    }
}

Tensor Tensor::matmul(const Tensor& other) const {
    assertValid("matmul(lhs)");
    other.assertValid("matmul(rhs)");
    
    if (!this->is_3d && !other.is_3d) {
        if (this->cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication");
        }

        const int M = this->rows;
        const int K = this->cols;
        const int N = other.cols;

        Tensor result(M, N);
        
        // Using OpenBLAS for matrix multiplication
        // C = alpha * A * B + beta * C
        // result = this * other, so alpha=1, beta=0
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K,
                    1.0f,          
                    this->data, K,     
                    other.data, N,       
                    0.0f,          
                    result.data, N);      
        
        return result;
        
    } else if (this->is_3d && !other.is_3d) {
        if (this->cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions do not match for batch multiplication");
        }
        
        const int batch_count = this->batch_size;
        const int M = this->rows;
        const int K = this->cols;
        const int N = other.cols;

        Tensor result(batch_count, M, N);

        for (int b = 0; b < batch_count; ++b) {
            const float* A = this->data + b * M * K;
            float* C = result.data + b * M * N;
            
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K,
                        1.0f,
                        A, K,
                        other.data, N,
                        0.0f,
                        C, N);
        }
        
        return result;
        
    } else if (this->is_3d && other.is_3d) {
        if (this->batch_size != other.batch_size || this->cols != other.rows) {
            throw std::invalid_argument("Batch matrix dimensions do not match");
        }
        
        const int batch_count = this->batch_size;
        const int M = this->rows;
        const int K = this->cols;
        const int N = other.cols;
        
        Tensor result(batch_count, M, N);
        
        for (int b = 0; b < batch_count; ++b) {
            const float* A = this->data + b * M * K;
            const float* B = other.data + b * K * N;
            float* C = result.data + b * M * N;
            
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K,
                        1.0f,
                        A, K,
                        B, N,
                        0.0f,
                        C, N);
        }
        
        return result;
        
    } else {
        throw std::invalid_argument("Unsupported matrix multiplication configuration");
    }
}

// Old Matrix Multuplication before using BLAS
/* Tensor Tensor::matmul(const Tensor& other) const {
    assertValid("matmul(lhs)");
    other.assertValid("matmul(rhs)");
    
    if (!this->is_3d && !other.is_3d) {
        if (this->cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication");
        }

        const int M = this->rows;
        const int K = this->cols;
        const int N = other.cols;

        Tensor result(M, N);
        const float* A = this->data;
        const float* B_data = other.data;
        float* C = result.raw();

        for (int i = 0; i < M * N; ++i) { C[i] = 0.0f; }

        const int Mc = 128;
        const int Nc = 128;
        const int Kc = 256;

        for (int i0 = 0; i0 < M; i0 += Mc) {
            const int i_max = (i0 + Mc < M) ? (i0 + Mc) : M;
            for (int k0 = 0; k0 < K; k0 += Kc) {
                const int k_max = (k0 + Kc < K) ? (k0 + Kc) : K;
                for (int j0 = 0; j0 < N; j0 += Nc) {
                    const int j_max = (j0 + Nc < N) ? (j0 + Nc) : N;

                    for (int i = i0; i < i_max; ++i) {
                        const float* Ai = A + i * K + k0;
                        float* Ci = C + i * N + j0;

                        for (int k = k0; k < k_max; ++k) {
                            const float a_ik = Ai[k - k0];
                            const float* Bk = B_data + k * N + j0;

                            for (int j = j0; j < j_max; ++j) {
                                Ci[j - j0] += a_ik * Bk[j - j0];
                            }
                        }
                    }
                }
            }
        }
        return result;
    } else if (this->is_3d && !other.is_3d) {
        if (this->cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions do not match for batch multiplication");
        }
        
        const int batch_count = this->batch_size;
        const int M = this->rows;
        const int K = this->cols;
        const int N = other.cols;

        Tensor result(batch_count, M, N);

        const int Mc = 64;
        const int Nc = 64;
        const int Kc = 128;

        const float* A_base = this->data;
        const float* B_data = other.data;
        float* C_base = result.data;

        for (int b = 0; b < batch_count; ++b) {
            const float* A = A_base + b * M * K;
            float* C = C_base + b * M * N;

            for (int i = 0; i < M * N; ++i) {
                C[i] = 0.0f;
            }

            for (int i0 = 0; i0 < M; i0 += Mc) {
                const int i_max = std::min(i0 + Mc, M);
                
                for (int k0 = 0; k0 < K; k0 += Kc) {
                    const int k_max = std::min(k0 + Kc, K);
                    
                    for (int j0 = 0; j0 < N; j0 += Nc) {
                        const int j_max = std::min(j0 + Nc, N);
                        
                        for (int i = i0; i < i_max; ++i) {
                            const float* A_row = A + i * K;
                            float* C_row = C + i * N;
                            
                            for (int k = k0; k < k_max; ++k) {
                                const float a_val = A_row[k];
                                const float* B_row = B_data + k * N;
                                
                                for (int j = j0; j < j_max; ++j) {
                                    C_row[j] += a_val * B_row[j];
                                }
                            }
                        }
                    }
                }
            }
        }
        return result;
    } else if (this->is_3d && other.is_3d) {
        if (this->batch_size != other.batch_size || this->cols != other.rows) {
            throw std::invalid_argument("Batch matrix dimensions do not match");
        }
        
        const int batch_count = this->batch_size;
        const int M = this->rows;
        const int K = this->cols;
        const int N = other.cols;
        
        Tensor result(batch_count, M, N);
        
        const int Mc = 64;
        const int Nc = 64;
        const int Kc = 128;
        
        const float* A_base = this->data;
        const float* B_base = other.data;
        float* C_base = result.data;
        
        for (int b = 0; b < batch_count; ++b) {
            const float* A = A_base + b * M * K;
            const float* B_ptr = B_base + b * K * N;
            float* C = C_base + b * M * N;
            
            for (int i = 0; i < M * N; ++i) {
                C[i] = 0.0f;
            }
            
            for (int i0 = 0; i0 < M; i0 += Mc) {
                const int i_max = std::min(i0 + Mc, M);
                
                for (int k0 = 0; k0 < K; k0 += Kc) {
                    const int k_max = std::min(k0 + Kc, K);
                    
                    for (int j0 = 0; j0 < N; j0 += Nc) {
                        const int j_max = std::min(j0 + Nc, N);
                        
                        for (int i = i0; i < i_max; ++i) {
                            const float* A_row = A + i * K;
                            float* C_row = C + i * N;
                            
                            for (int k = k0; k < k_max; ++k) {
                                const float a_val = A_row[k];
                                const float* B_row = B_ptr + k * N;
                                
                                for (int j = j0; j < j_max; ++j) {
                                    C_row[j] += a_val * B_row[j];
                                }
                            }
                        }
                    }
                }
            }
        }
        return result;
    } else {
        throw std::invalid_argument("Unsupported matrix multiplication configuration");
    }
} */

Tensor Tensor::add(const Tensor& other) const {
    assertValid("add(lhs)");
    other.assertValid("add(rhs)");
    if (!this->is_3d && !other.is_3d) {
        bool rows_compatible = (rows == other.rows) || (rows == 1) || (other.rows == 1);
        bool cols_compatible = (cols == other.cols) || (cols == 1) || (other.cols == 1);
        if (!rows_compatible || !cols_compatible) {
            throw std::invalid_argument("Shapes not broadcastable");
        }

        const int R = std::max(rows, other.rows);
        const int C = std::max(cols, other.cols);
        Tensor result(R, C);

        const float* A = this->data;
        const float* B = other.data;
        float* Out = result.raw();

        const bool a_row_bcast = (rows == 1);
        const bool a_col_bcast = (cols == 1);
        const bool b_row_bcast = (other.rows == 1);
        const bool b_col_bcast = (other.cols == 1);

        for (int i = 0; i < R; ++i) {
            const int ai = a_row_bcast ? 0 : i;
            const int bi = b_row_bcast ? 0 : i;
            const int a_row_off = ai * cols;
            const int b_row_off = bi * other.cols;
            const int out_row_off = i * C;

            for (int j = 0; j < C; ++j) {
                const int aj = a_col_bcast ? 0 : j;
                const int bj = b_col_bcast ? 0 : j;
                Out[out_row_off + j] = A[a_row_off + aj] + B[b_row_off + bj];
            }
        }
        return result;
    } else if (this->is_3d && !other.is_3d) {
        bool rows_compatible = (rows == other.rows) || (rows == 1) || (other.rows == 1);
        bool cols_compatible = (cols == other.cols) || (cols == 1) || (other.cols == 1);
        if (!rows_compatible || !cols_compatible) {
            throw std::invalid_argument("Tensor dimensions don't match for broadcasting");
        }

        Tensor result(batch_size, rows, cols);
        const float* A = this->raw();
        const float* B = other.raw();
        float* C = result.raw();
        
        const bool b_row_bcast = (other.rows == 1);
        const bool b_col_bcast = (other.cols == 1);
        
        for (int b = 0; b < batch_size; ++b) {
            const int batch_offset = b * rows * cols;
            for (int i = 0; i < rows; ++i) {
                const int oi = b_row_bcast ? 0 : i;
                const int row_offset = batch_offset + i * cols;
                const int other_row_offset = oi * other.cols;
                
                for (int j = 0; j < cols; ++j) {
                    const int oj = b_col_bcast ? 0 : j;
                    C[row_offset + j] = A[row_offset + j] + B[other_row_offset + oj];
                }
            }
        }
        return result;
        
    } else if (!this->is_3d && other.is_3d) {
        bool rows_compatible = (this->rows == other.rows) || (this->rows == 1) || (other.rows == 1);
        bool cols_compatible = (this->cols == other.cols) || (this->cols == 1) || (other.cols == 1);
        if (!rows_compatible || !cols_compatible) {
            throw std::invalid_argument("Tensor dimensions don't match for broadcasting");
        }

        Tensor result(other.batch_size, other.rows, other.cols);
        const float* A = this->raw();
        const float* B = other.raw();
        float* C = result.raw();
        
        const bool a_row_bcast = (this->rows == 1);
        const bool a_col_bcast = (this->cols == 1);
        
        for (int b = 0; b < other.batch_size; b++) {
            const int batch_offset = b * other.rows * other.cols;
            for (int i = 0; i < other.rows; i++) {
                const int this_i = a_row_bcast ? 0 : i;
                const int row_offset = batch_offset + i * other.cols;
                const int this_row_offset = this_i * this->cols;
                
                for (int j = 0; j < other.cols; j++) {
                    const int this_j = a_col_bcast ? 0 : j;
                    C[row_offset + j] = A[this_row_offset + this_j] + B[row_offset + j];
                }
            }
        }
        return result;
        
    } else if (this->is_3d && other.is_3d) {
        if (this->batch_size != other.batch_size || this->rows != other.rows || this->cols != other.cols) {
            throw std::invalid_argument("3D tensor dimensions don't match");
        }
        
        Tensor result(this->batch_size, this->rows, this->cols);
        const float* A = this->raw();
        const float* B = other.raw();
        float* C = result.raw();
        const int total = this->batch_size * this->rows * this->cols;
        
        blas_vadd(A, B, C, total);
        return result;
    } else {
        throw std::invalid_argument("Unsupported addition configuration");
    }
}

Tensor Tensor::subtract(const Tensor& other) const {
    assertValid("subtract(lhs)");
    other.assertValid("subtract(rhs)");

    if (!this->is_3d && !other.is_3d) {
        if (this->rows != other.rows || this->cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions do not match for subtraction");
        }
        Tensor result(this->rows, this->cols);
        const float* A = this->raw();
        const float* B = other.raw();
        float* C = result.raw();
        const int total = this->rows * this->cols;
        
        blas_vsub(other.data, data, result.data, total);
        
        return result;
    } else if (this->is_3d && other.is_3d) {
        if (this->batch_size != other.batch_size || this->rows != other.rows || this->cols != other.cols) {
            throw std::invalid_argument("3D tensor dimensions don't match for subtraction");
        }
        Tensor result(this->batch_size, this->rows, this->cols);
        const float* A = this->raw();
        const float* B = other.raw();
        float* C = result.raw();
        const int total = this->batch_size * this->rows * this->cols;

        blas_vsub(other.data, data, result.data, total);

        return result;
    } else {
        throw std::invalid_argument("Cannot subtract tensors with different dimensionalities");
    }
}

Tensor Tensor::elementwise(const Tensor& other) const {
    assertValid("elementwise(lhs)");
    other.assertValid("elementwise(rhs)");
    
    if (!this->is_3d && !other.is_3d) {
        if (this->rows != other.rows || this->cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions do not match for elementwise multiply");
        }
        Tensor result(this->rows, this->cols);
        const float* A = this->raw();
        const float* B = other.raw();
        float* C = result.raw();
        const int total = this->rows * this->cols;
        
        blas_vmul(data, other.data, result.data, total);
        
        return result;
        
    } else if (this->is_3d && other.is_3d) {
        if (this->batch_size != other.batch_size || this->rows != other.rows || this->cols != other.cols) {
            throw std::invalid_argument("3D tensor dimensions don't match for elementwise multiply");
        }
        Tensor result(this->batch_size, this->rows, this->cols);
        const float* A = this->raw();
        const float* B = other.raw();
        float* C = result.raw();
        const int total = this->batch_size * this->rows * this->cols;
        
        blas_vmul(A, B, C, total);
        return result;
    } else {
        throw std::invalid_argument("Cannot perform elementwise multiply on tensors with different dimensionalities");
    }
}

Tensor Tensor::transpose() const {
    assertValid("transpose(this)");
    
    if (!this->is_3d) {
        Tensor result(this->cols, this->rows);
        const float* src = this->raw();
        float* dst = result.raw();
        
        const int BLOCK = 32;
        
        for (int i0 = 0; i0 < this->rows; i0 += BLOCK) {
            const int i_max = std::min(i0 + BLOCK, this->rows);
            for (int j0 = 0; j0 < this->cols; j0 += BLOCK) {
                const int j_max = std::min(j0 + BLOCK, this->cols);
                
                for (int i = i0; i < i_max; ++i) {
                    for (int j = j0; j < j_max; ++j) {
                        dst[j * this->rows + i] = src[i * this->cols + j];
                    }
                }
            }
        }
        return result;
        
    } else {
        Tensor result(this->batch_size, this->cols, this->rows);
        const float* src = this->raw();
        float* dst = result.raw();
        
        const int BLOCK = 32;
        
        for (int b = 0; b < this->batch_size; ++b) {
            const float* batch_src = src + b * this->rows * this->cols;
            float* batch_dst = dst + b * this->cols * this->rows;
            
            for (int i0 = 0; i0 < this->rows; i0 += BLOCK) {
                const int i_max = std::min(i0 + BLOCK, this->rows);
                for (int j0 = 0; j0 < this->cols; j0 += BLOCK) {
                    const int j_max = std::min(j0 + BLOCK, this->cols);
                    
                    for (int i = i0; i < i_max; ++i) {
                        for (int j = j0; j < j_max; ++j) {
                            batch_dst[j * this->rows + i] = batch_src[i * this->cols + j];
                        }
                    }
                }
            }
        }
        return result;
    }
}

Tensor Tensor::softmax() const {
    assertValid("softmax(this)");

    if (!this->is_3d) {
        Tensor result(this->rows, this->cols);
        const float* input_data = this->raw();
        float* output_data = result.raw();

        for (int i = 0; i < this->rows; i++) {
            const float* row_in = input_data + i * this->cols;
            float* row_out = output_data + i * this->cols;

            float max_val = row_in[0];
            for (int j = 1; j < this->cols; j++) {
                max_val = std::max(max_val, row_in[j]);
            }

            float sum = 0.0f;
            for (int j = 0; j < this->cols; j++) {
                row_out[j] = std::expf(row_in[j] - max_val);
                sum += row_out[j];
            }

            if (sum <= 0.0f) throw std::runtime_error("softmax sum <= 0 (numerical underflow)");

            const float inv_sum = 1.0f / sum;
            for (int j = 0; j < this->cols; j++) {
                row_out[j] *= inv_sum;
            }
        }
        return result;
    } else {
        Tensor result(this->batch_size, this->rows, this->cols);
        const float* input_data = this->raw();
        float* output_data = result.raw();

        for (int b = 0; b < this->batch_size; b++) {
            int batch_offset = b * this->rows * this->cols;

            for (int i = 0; i < this->rows; i++) {
                const float* row_in = input_data + batch_offset + i * this->cols;
                float* row_out = output_data + batch_offset + i * this->cols;
                
                float max_val = row_in[0];
                for (int j = 1; j < this->cols; j++) {
                    max_val = std::max(max_val, row_in[j]);
                }

                float sum = 0.0f;
                for (int j = 0; j < this->cols; j++) {
                    row_out[j] = std::expf(row_in[j] - max_val);
                    sum += row_out[j];
                }

                if (sum <= 0.0f) throw std::runtime_error("softmax sum <= 0 (numerical underflow)");

                const float inv_sum = 1.0f / sum;
                for (int j = 0; j < this->cols; j++) {
                    row_out[j] *= inv_sum;
                }
            }
        }
        return result;
    }
}

void Tensor::fill(float value) {
    assertValid("fill(this)");
    const int total = batch_size * rows * cols;
    blas_vfill(value, data, total);
}

Tensor Tensor::scale(float scaler) const {
    assertValid("scale(this)");

    if (!this->is_3d) {
        Tensor result(this->rows, this->cols);
        const float* src = this->raw();
        float* dst = result.raw();
        const int total = this->rows * this->cols;

        blas_vsmul(src, scaler, dst, total);
        return result;
    } else {
        Tensor result(this->batch_size, this->rows, this->cols);
        const float* src = this->raw();
        float* dst = result.raw();
        const int total = this->batch_size * this->rows * this->cols;

        blas_vsmul(src, scaler, dst, total);
        return result;
    }
}

Tensor Tensor::reshape(int new_rows, int new_cols) const {
    assertValid("reshape(this)");
    if (is_3d) throw std::invalid_argument("reshape: 3D not supported yet");
    if (new_rows * new_cols != rows * cols) {
        throw std::invalid_argument("Matrix sizes do not match for reshape");
    }
    Tensor result(new_rows, new_cols);
    float* out = result.raw();
    const float* in = data;
    for (int i = 0; i < new_rows * new_cols; ++i) out[i] = in[i];
    return result;
}

Tensor Tensor::slice(int start_row, int num_rows, int start_col, int num_cols) const {
    assertValid("slice(this)");
    if (is_3d) throw std::invalid_argument("slice: 3D not supported yet");

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
    assertValid("concatenate(lhs)");
    other.assertValid("concatenate(rhs)");
    if (is_3d || other.is_3d) {
        throw std::invalid_argument("concatenate: 3D not supported yet");
    }
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
    assertValid("xavier(target)");
    static std::random_device rd;
    static std::mt19937 gen(rd());

    float limit = std::sqrt(6.0f / (fan_in + fan_out));
    std::uniform_real_distribution<float> dis(-limit, limit);

    for (int i = 0; i < batch_size * rows * cols; i++) {
        data[i] = dis(gen);
    }
}

Tensor Tensor::create_causal_mask(int seq_len) {
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

Tensor Tensor::create_causal_mask_batch(int batch_size, int seq_len) {
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

void Tensor::assertValid(const std::string& context) const {
    if (data == nullptr) {
        throw std::runtime_error("Tensor error [" + context + "]: data pointer is null");
    }
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Tensor error [" + context + "]: invalid shape (" +
                                 std::to_string(rows) + "x" + std::to_string(cols) + ")");
    }
    if (is_3d && batch_size <= 0) {
        throw std::runtime_error("Tensor error [" + context + "]: invalid batch_size " +
                                 std::to_string(batch_size));
    }

    const int expected = is_3d ? batch_size * rows * cols : rows * cols;
    if (expected <= 0) {
        throw std::runtime_error("Tensor error [" + context + "]: zero elements");
    }
}

void Tensor::scale_inplace(float scalar) {
    assertValid("scale_inplace");
    const int total = batch_size * rows * cols;
    blas_vsmul(data, scalar, data, total);
}

void Tensor::add_inplace(const Tensor& other) {
    assertValid("add_inplace");
    other.assertValid("add_inplace(other)");
    
    if (!is_3d && !other.is_3d) {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Shape mismatch for in-place add");
        }
        
        const int total = rows * cols;
        const float* other_data = other.raw();
        blas_vadd(data, other_data, data, total);

    } else if (is_3d && other.is_3d) {
        if (batch_size != other.batch_size || rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Shape mismatch for in-place add");
        }
        
        const int total = batch_size * rows * cols;
        const float* other_data = other.raw();
        blas_vadd(data, other_data, data, total);

    } else {
        throw std::invalid_argument("Cannot add 2D and 3D tensors in-place");
    }
}

void Tensor::multiply_inplace(const Tensor& other) {
    assertValid("multiply_inplace");
    other.assertValid("multiply_inplace(other)");
    
    if (rows != other.rows || cols != other.cols || is_3d != other.is_3d) {
        throw std::invalid_argument("Shape mismatch for in-place multiply");
    }
    
    const int total = (is_3d ? batch_size : 1) * rows * cols;
    const float* other_data = other.raw();
    blas_vmul(data, other_data, data, total);

}

void Tensor::zero() {
    const int total = (is_3d ? batch_size : 1) * rows * cols;
    std::memset(data, 0, total * sizeof(float));
}