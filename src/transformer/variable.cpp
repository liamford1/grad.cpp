#include "transformer/variable.h"
#include "transformer/activations.h"
#include "transformer/blas_wrapper.h"
#include "transformer/parallel.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

// Grads stay empty until ensureGrad() - see the header note on lazy
// gradient allocation.
Variable::Variable(Private, const Tensor& data, bool requires_grad)
    : data(data), requires_grad(requires_grad) {}

Variable::Variable(Private, Tensor&& data, bool requires_grad)
    : data(std::move(data)), requires_grad(requires_grad) {}

Variable::Variable(Private, int rows, int cols, bool requires_grad)
    : data(rows, cols), requires_grad(requires_grad) {}

Variable::Variable(Private, int batch_size, int rows, int cols, bool requires_grad)
    : data(batch_size, rows, cols), requires_grad(requires_grad) {}

void Variable::ensureGrad() {
    if (!requires_grad || grad.numel() > 0) return;
    // The Tensor constructor zero-fills, so the grad is accumulation-ready.
    grad = data.getIs3D()
        ? Tensor(data.getBatchSize(), data.getRows(), data.getCols())
        : Tensor(data.getRows(), data.getCols());
}

std::shared_ptr<Variable> Variable::create(const Tensor& data, bool requires_grad) {
    return std::make_shared<Variable>(Private{}, data, requires_grad);
}

std::shared_ptr<Variable> Variable::create(Tensor&& data, bool requires_grad) {
    return std::make_shared<Variable>(Private{}, std::move(data), requires_grad);
}

std::shared_ptr<Variable> Variable::create(int rows, int cols, bool requires_grad) {
    return std::make_shared<Variable>(Private{}, rows, cols, requires_grad);
}

std::shared_ptr<Variable> Variable::create(int batch_size, int rows, int cols, bool requires_grad) {
    return std::make_shared<Variable>(Private{}, batch_size, rows, cols, requires_grad);
}

std::shared_ptr<Variable> Variable::createOutput(Tensor&& result, bool needs_grad) {
    return std::make_shared<Variable>(Private{}, std::move(result), needs_grad);
}

std::shared_ptr<Variable> Variable::matmul(std::shared_ptr<Variable> other) {
    data.assertValid("Variable::matmul(lhs)");
    other->data.assertValid("Variable::matmul(rhs)");

    Tensor result = this->data.matmul(other->data);
    bool needs_grad = this->requires_grad || other->requires_grad;
    
    auto output = createOutput(std::move(result), needs_grad);
    
    if (needs_grad) {
        auto self_ptr = shared_from_this();
        
        output->addChild(self_ptr);
        output->addChild(other);
        
        output->setBackwardFn([self_ptr, other, output_weak = std::weak_ptr<Variable>(output)]() {
            auto output = output_weak.lock();
            if (!output || !output->hasGrad()) return;
            self_ptr->data.assertValid("Variable::matmul(self.data)");
            other->data.assertValid("Variable::matmul(other.data)");

            if (!other->data.getIs3D()) {
                // X (flat, K) @ W (K, N) = Y (flat, N), where a 3D X is its
                // contiguous (batch*rows, K) view. Both gradients are single
                // sgemms accumulated in place (beta = 1); the sgemm's transA
                // sums dW over batch*rows with no temporaries.
                const Tensor& X = self_ptr->data;
                const Tensor& dY = output->grad;
                int K = other->data.getRows();
                int N = other->data.getCols();
                int flat = X.getIs3D() ? X.getBatchSize() * X.getRows() : X.getRows();

                if (self_ptr->requires_grad) {
                    self_ptr->ensureGrad();
                    // dX += dY @ W^T
                    blas_sgemm_ex(dY.raw(), other->data.raw(), self_ptr->grad.raw(),
                                  flat, K, N, false, true, 1.0f, 1.0f);
                }
                if (other->requires_grad) {
                    other->ensureGrad();
                    // dW += X^T @ dY
                    blas_sgemm_ex(X.raw(), dY.raw(), other->grad.raw(),
                                  K, N, flat, true, false, 1.0f, 1.0f);
                }
            } else {
                if (self_ptr->requires_grad) {
                    self_ptr->ensureGrad();
                    Tensor other_transposed = other->data.transpose();
                    Tensor self_grad = output->grad.matmul(other_transposed);
                    self_ptr->grad.add_inplace(self_grad);
                }
                if (other->requires_grad) {
                    other->ensureGrad();
                    Tensor self_transposed = self_ptr->data.transpose();
                    Tensor other_grad = self_transposed.matmul(output->grad);
                    other->grad.add_inplace(other_grad);
                }
            }
        });
    }
    return output;
}

std::shared_ptr<Variable> Variable::add(std::shared_ptr<Variable> other) {
    data.assertValid("Variable::add(lhs)");
    other->data.assertValid("Variable::add(rhs)");

    Tensor result = this->data.add(other->data);
    bool needs_grad = this->requires_grad || other->requires_grad;
    auto output = createOutput(std::move(result), needs_grad);

    if (needs_grad) {
        auto self_ptr = shared_from_this();

        output->addChild(self_ptr);
        output->addChild(other);

        output->setBackwardFn([self_ptr, other, output_weak = std::weak_ptr<Variable>(output)]() {
            auto output = output_weak.lock();
            if (!output || !output->hasGrad()) return;

            const Tensor& x  = self_ptr->data;
            const Tensor& dO = output->grad;

            auto reduce2D = [](const Tensor& g, int R, int C, bool br, bool bc) -> Tensor {
                if (!br && !bc && g.getRows() == static_cast<size_t>(R) && g.getCols() == static_cast<size_t>(C) && !g.getIs3D()) {
                    Tensor out(R, C);
                    const float* src = g.raw();
                    float* dst = out.raw();
                    const int total = R * C;
                    for (int i = 0; i < total; ++i) dst[i] = src[i];
                    return out;
                }

                Tensor out(R, C);
                out.fill(0.0f);
                const int GR = g.getRows();
                const int GC = g.getCols();
                
                const float* g_ptr = g.raw();
                float* out_ptr = out.raw();

                if (br && bc) {
                    float s = 0.0f;
                    const int total = GR * GC;
                    for (int i = 0; i < total; ++i) {
                        s += g_ptr[i];
                    }
                    out_ptr[0] = s;
                } else if (br) {
                    for (int j = 0; j < C; ++j) {
                        float s = 0.0f;
                        for (int ii = 0; ii < GR; ++ii) {
                            s += g_ptr[ii * GC + j];
                        }
                        out_ptr[j] = s;
                    }
                } else if (bc) {
                    for (int i = 0; i < R; ++i) {
                        float s = 0.0f;
                        const float* g_row = g_ptr + i * GC;
                        for (int jj = 0; jj < GC; ++jj) {
                            s += g_row[jj];
                        }
                        out_ptr[i] = s;
                    }
                } else {
                    for (int i = 0; i < R; ++i) {
                        for (int j = 0; j < C; ++j) {
                            out_ptr[i * C + j] = g_ptr[i * GC + j];
                        }
                    }
                }
                return out;
            };

            auto reduce3Dfrom2D = [](const Tensor& g3, int R, int C, bool br, bool bc) -> Tensor {
                Tensor out(R, C);
                out.fill(0.0f);
                const int B  = g3.getBatchSize();
                const int GR = g3.getRows();
                const int GC = g3.getCols();

                const float* g3_ptr = g3.raw();
                float* out_ptr = out.raw();

                if (!br && !bc) {
                    for (int b = 0; b < B; ++b) {
                        const float* batch_ptr = g3_ptr + b * GR * GC;
                        for (int i = 0; i < R; ++i) {
                            const float* row_ptr = batch_ptr + i * GC;
                            float* out_row = out_ptr + i * C;
                            for (int j = 0; j < C; ++j) {
                                out_row[j] += row_ptr[j];
                            }
                        }
                    }
                    return out;
                }

                if (br && bc) {
                    float s = 0.0f;
                    const int total = B * GR * GC;
                    for (int i = 0; i < total; ++i) {
                        s += g3_ptr[i];
                    }
                    out_ptr[0] = s;
                } else if (br) {
                    for (int j = 0; j < C; ++j) {
                        float s = 0.0f;
                        for (int b = 0; b < B; ++b) {
                            for (int ii = 0; ii < GR; ++ii) {
                                s += g3_ptr[b * GR * GC + ii * GC + j];
                            }
                        }
                        out_ptr[j] = s;
                    }
                } else {
                    for (int i = 0; i < R; ++i) {
                        float s = 0.0f;
                        for (int b = 0; b < B; ++b) {
                            const float* batch_row = g3_ptr + b * GR * GC + i * GC;
                            for (int jj = 0; jj < GC; ++jj) {
                                s += batch_row[jj];
                            }
                        }
                        out_ptr[i] = s;
                    }
                }
                return out;
            };

            // Fast paths for the two shapes that dominate training:
            // same-shape adds (residual connections) accumulate directly,
            // and row-vector biases reduce via cache-friendly row-major
            // column sums. Everything else falls back to the general
            // broadcast reduction below.
            auto fast_accumulate = [&dO](const Tensor& shape, Tensor& grad) -> bool {
                const bool same_shape = shape.getIs3D() == dO.getIs3D()
                    && shape.getRows() == dO.getRows()
                    && shape.getCols() == dO.getCols()
                    && (!shape.getIs3D() || shape.getBatchSize() == dO.getBatchSize());
                if (same_shape) {
                    blas_vadd(grad.raw(), dO.raw(), grad.raw(), dO.numel());
                    return true;
                }
                if (!shape.getIs3D() && shape.getRows() == 1
                    && shape.getCols() == dO.getCols() && dO.getCols() > 1) {
                    const size_t rows = dO.getIs3D()
                        ? dO.getBatchSize() * dO.getRows() : dO.getRows();
                    const size_t C = dO.getCols();
                    const float* g = dO.raw();
                    float* out = grad.raw();
                    for (size_t i = 0; i < rows; i++) {
                        const float* row = g + i * C;
                        for (size_t j = 0; j < C; j++) {
                            out[j] += row[j];
                        }
                    }
                    return true;
                }
                return false;
            };

            if (self_ptr->requires_grad) self_ptr->ensureGrad();
            if (other->requires_grad) other->ensureGrad();

            if (self_ptr->requires_grad && !fast_accumulate(x, self_ptr->grad)) {
                if (!x.getIs3D() && !dO.getIs3D()) {
                    bool br = (x.getRows() == 1) && (dO.getRows() > 1);
                    bool bc = (x.getCols() == 1) && (dO.getCols() > 1);
                    Tensor dx = reduce2D(dO, x.getRows(), x.getCols(), br, bc);
                    self_ptr->grad.add_inplace(dx);
                } else if (!x.getIs3D() && dO.getIs3D()) {
                    bool br = (x.getRows() == 1) && (dO.getRows() > 1);
                    bool bc = (x.getCols() == 1) && (dO.getCols() > 1);
                    Tensor dx = reduce3Dfrom2D(dO, x.getRows(), x.getCols(), br, bc);
                    self_ptr->grad.add_inplace(dx);
                } else {
                    throw std::runtime_error("add backward: unexpected shape combination for x");
                }
            }

            if (other->requires_grad && !fast_accumulate(other->data, other->grad)) {
                const Tensor& yD = other->data;
                if (!yD.getIs3D() && !dO.getIs3D()) {
                    bool br = (yD.getRows() == 1) && (dO.getRows() > 1);
                    bool bc = (yD.getCols() == 1) && (dO.getCols() > 1);
                    Tensor dy = reduce2D(dO, yD.getRows(), yD.getCols(), br, bc);
                    other->grad.add_inplace(dy);
                } else if (!yD.getIs3D() && dO.getIs3D()) {
                    bool br = (yD.getRows() == 1) && (dO.getRows() > 1);
                    bool bc = (yD.getCols() == 1) && (dO.getCols() > 1);
                    Tensor dy = reduce3Dfrom2D(dO, yD.getRows(), yD.getCols(), br, bc);
                    other->grad.add_inplace(dy);
                } else {
                    throw std::runtime_error("add backward: unexpected shape combination for y");
                }
            }
        });
    }
    return output;
}


std::shared_ptr<Variable> Variable::scale(float factor) {
    data.assertValid("Variable::scale(x)");

    Tensor result = this->data.scale(factor);
    auto output = createOutput(std::move(result), this->requires_grad);
    
    if (this->requires_grad) {
        auto self_ptr = shared_from_this();
        
        output->addChild(self_ptr);
        output->setBackwardFn([self_ptr, factor, output_weak = std::weak_ptr<Variable>(output)]() {
            auto output = output_weak.lock();
            if (!output || !output->hasGrad()) return;
            if (self_ptr->requires_grad) {
                self_ptr->ensureGrad();
                Tensor scaled_grad = output->grad.scale(factor);
                self_ptr->grad.add_inplace(scaled_grad);
            }
        });
    }
    return output;
}

std::shared_ptr<Variable> Variable::softmax() {
    data.assertValid("Variable::softmax(x)");

    Tensor result = this->data.softmax();
    auto output = createOutput(std::move(result), this->requires_grad);
    
    if (this->requires_grad) {
        auto self_ptr = shared_from_this();
        
        output->addChild(self_ptr);
        // The softmax output needed by backward IS this node's data; the
        // retire-as-you-go backward frees it only after this fn has run,
        // so reading it here avoids capturing a full copy.
        output->setBackwardFn([self_ptr, output_weak = std::weak_ptr<Variable>(output)]() {
            auto output = output_weak.lock();
            if (!output || !output->hasGrad()) return;
            const Tensor& result = output->getData();
            result.assertValid("Variable::softmax(y)");

            if (self_ptr->requires_grad) {
                self_ptr->ensureGrad();
                if (result.getIs3D()) {
                    Tensor temp_grad(result.getBatchSize(), result.getRows(), result.getCols());
                    const float* result_data = result.raw();
                    const float* grad_out_data = output->grad.raw();
                    float* temp_grad_data = temp_grad.raw();

                    for (size_t b = 0; b < result.getBatchSize(); b++) {
                        const size_t batch_offset = b * result.getRows() * result.getCols();

                        for (size_t i = 0; i < result.getRows(); i++) {
                            const float* row_result = result_data + batch_offset + i * result.getCols();
                            const float* row_grad_out = grad_out_data + batch_offset + i * result.getCols();
                            float* row_grad = temp_grad_data + batch_offset + i * result.getCols();

                            float dot_product = 0.0f;
                            for (size_t j = 0; j < result.getCols(); j++) {
                                dot_product += row_result[j] * row_grad_out[j];
                            }

                            for (size_t j = 0; j < result.getCols(); j++) {
                                row_grad[j] = row_result[j] * (row_grad_out[j] - dot_product);
                            }
                        }
                    }
                    self_ptr->grad.add_inplace(temp_grad);
                } else {
                    Tensor temp_grad(result.getRows(), result.getCols());
                    const float* result_data = result.raw();
                    const float* grad_out_data = output->grad.raw();
                    float* temp_grad_data = temp_grad.raw();

                    for (size_t i = 0; i < result.getRows(); i++) {
                        const float* row_result = result_data + i * result.getCols();
                        const float* row_grad_out = grad_out_data + i * result.getCols();
                        float* row_grad = temp_grad_data + i * result.getCols();

                        float dot_product = 0.0f;
                        for (size_t j = 0; j < result.getCols(); j++) {
                            dot_product += row_result[j] * row_grad_out[j];
                        }

                        for (size_t j = 0; j < result.getCols(); j++) {
                            row_grad[j] = row_result[j] * (row_grad_out[j] - dot_product);
                        }
                    }
                    self_ptr->grad.add_inplace(temp_grad);
                }
            }
        });
    }
    return output;
}

std::shared_ptr<Variable> Variable::cross_entropy_loss(std::shared_ptr<Variable> targets) {
    data.assertValid("Variable::cross_entropy_loss(input)");
    targets->data.assertValid("Variable::cross_entropy_loss(targets)");
    
    if (!this->data.getIs3D() && !targets->data.getIs3D()) {
        Tensor loss_tensor(1, 1);
        float total_loss = 0.0f;
        
        if (targets->data.getCols() == 1) {
            for (size_t i = 0; i < this->data.getRows(); i++) {
                int target_idx = static_cast<int>(targets->data.getValue(i, 0));
                if (target_idx >= 0 && static_cast<size_t>(target_idx) < this->data.getCols()) {
                    float prob = std::max(this->data.getValue(i, target_idx), 1e-15f);
                    total_loss -= std::log(prob);
                }
            }
        } else {
            for (size_t i = 0; i < this->data.getRows(); i++) {
                for (size_t j = 0; j < this->data.getCols(); j++) {
                    if (targets->data.getValue(i, j) > 0.0f) {
                        float prob = std::max(this->data.getValue(i, j), 1e-15f);
                        total_loss -= targets->data.getValue(i, j) * std::log(prob);
                    }
                }
            }
        }
        total_loss /= this->data.getRows();
        loss_tensor.setValue(0, 0, total_loss);
        auto output = createOutput(std::move(loss_tensor), this->requires_grad || targets->requires_grad);
        
        if (output->requires_grad) {
            auto self_ptr = shared_from_this();
            output->addChild(self_ptr);
            output->addChild(targets);
            output->setBackwardFn([self_ptr, targets, output_weak = std::weak_ptr<Variable>(output)]() {
                auto output = output_weak.lock();
                if (!output) return;
                if (self_ptr->requires_grad) {
                    self_ptr->ensureGrad();
                    if (targets->data.getCols() == 1) {
                        Tensor grad_tensor(self_ptr->data.getRows(), self_ptr->data.getCols());
                        grad_tensor.fill(0.0f);
                        float scale = 1.0f / self_ptr->data.getRows();

                        for (size_t i = 0; i < self_ptr->data.getRows(); i++) {
                            int target_idx = static_cast<int>(targets->data.getValue(i, 0));
                            if (target_idx >= 0 && static_cast<size_t>(target_idx) < self_ptr->data.getCols()) {
                                for (size_t j = 0; j < self_ptr->data.getCols(); j++) {
                                    float grad_val = self_ptr->data.getValue(i, j) * scale;
                                    if (j == static_cast<size_t>(target_idx)) {
                                        grad_val -= scale;
                                    }
                                    grad_tensor.setValue(i, j, grad_val);
                                }
                            }
                        }
                        self_ptr->grad.add_inplace(grad_tensor);
                    } else {
                        Tensor diff = self_ptr->data.subtract(targets->data);
                        Tensor scaled_diff = diff.scale(1.0f / self_ptr->data.getRows());
                        self_ptr->grad.add_inplace(scaled_diff);
                    }
                }
            });
        }
        return output;
        
    } else if (this->data.getIs3D()) {
        Tensor loss_tensor(1, 1);
        float total_loss = 0.0f;
        int batch_size = this->data.getBatchSize();
        int seq_len = this->data.getRows();
        int total_elements = batch_size * seq_len;
        
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < seq_len; i++) {
                int target_idx = targets->data.getIs3D() ? static_cast<int>(targets->data.getValue(b, i, 0)) : static_cast<int>(targets->data.getValue(b, i));

                if (target_idx >= 0 && static_cast<size_t>(target_idx) < this->data.getCols()) {
                    float prob = std::max(this->data.getValue(b, i, target_idx), 1e-15f);
                    total_loss -= std::log(prob);
                }
            }
        }
        
        total_loss /= total_elements;
        loss_tensor.setValue(0, 0, total_loss);
        auto output = createOutput(std::move(loss_tensor), this->requires_grad || targets->requires_grad);
        
        if (output->requires_grad) {
            auto self_ptr = shared_from_this();
            output->addChild(self_ptr);
            output->addChild(targets);
            
            output->setBackwardFn([self_ptr, targets, batch_size, seq_len, total_elements, output_weak = std::weak_ptr<Variable>(output)]() {
                auto output = output_weak.lock();
                if (!output) return;
                if (self_ptr->requires_grad) {
                    self_ptr->ensureGrad();
                    Tensor grad_tensor(batch_size, seq_len, self_ptr->data.getCols());
                    grad_tensor.fill(0.0f);
                    float scale = 1.0f / total_elements;

                    for (int b = 0; b < batch_size; b++) {
                        for (int i = 0; i < seq_len; i++) {
                            int target_idx = targets->data.getIs3D() ? static_cast<int>(targets->data.getValue(b, i, 0)) : static_cast<int>(targets->data.getValue(b, i));

                            if (target_idx >= 0 && static_cast<size_t>(target_idx) < self_ptr->data.getCols()) {
                                for (size_t j = 0; j < self_ptr->data.getCols(); j++) {
                                    float grad_val = self_ptr->data.getValue(b, i, j) * scale;
                                    if (j == static_cast<size_t>(target_idx)) {
                                        grad_val -= scale;
                                    }
                                    grad_tensor.setValue(b, i, j, grad_val);
                                }
                            }
                        }
                    }
                    self_ptr->grad.add_inplace(grad_tensor);
                }
            });
        }
        return output;
    } else {
        throw std::runtime_error("Unsupported tensor configuration for cross-entropy loss");
    }
}

std::shared_ptr<Variable> Variable::gelu() {
    data.assertValid("Variable::gelu(x)");

    // tanh approximation: gelu(x) = 0.5x(1 + tanh(k(x + a*x^3))).
    // The tanh runs through vec_tanh (SIMD); the polynomial loops
    // auto-vectorize.
    constexpr float k = 0.79788456f;
    constexpr float a = 0.044715f;
    const size_t n = data.numel();
    const float* x = data.raw();

    Tensor result = data.getIs3D()
        ? Tensor::uninitialized(data.getBatchSize(), data.getRows(), data.getCols())
        : Tensor::uninitialized(data.getRows(), data.getCols());
    float* out = result.raw();

    parallel_for(n, 32768, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; i++) {
            out[i] = k * (x[i] + a * x[i] * x[i] * x[i]);
        }
        vec_tanh(out + begin, out + begin, static_cast<int>(end - begin));
        for (size_t i = begin; i < end; i++) {
            out[i] = 0.5f * x[i] * (1.0f + out[i]);
        }
    });

    auto output = createOutput(std::move(result), this->requires_grad);

    if (this->requires_grad) {
        auto self_ptr = shared_from_this();
        output->addChild(self_ptr);
        output->setBackwardFn([self_ptr, output_weak = std::weak_ptr<Variable>(output)]() {
            auto output = output_weak.lock();
            if (!output || !output->hasGrad()) return;
            if (!self_ptr->requires_grad) return;
            self_ptr->ensureGrad();

            const size_t n = self_ptr->data.numel();
            const float* x = self_ptr->data.raw();
            const float* dY = output->grad.raw();
            float* dX = self_ptr->grad.raw();

            parallel_for(n, 32768, [&](size_t begin, size_t end) {
                const size_t len = end - begin;
                std::vector<float> t(len);
                for (size_t i = 0; i < len; i++) {
                    float xi = x[begin + i];
                    t[i] = k * (xi + a * xi * xi * xi);
                }
                vec_tanh(t.data(), t.data(), static_cast<int>(len));

                for (size_t i = 0; i < len; i++) {
                    float xi = x[begin + i];
                    float tv = t[i];
                    float sech_sq = 1.0f - tv * tv;
                    float dgelu = 0.5f * (1.0f + tv
                        + xi * sech_sq * k * (1.0f + 3.0f * a * xi * xi));
                    dX[begin + i] += dgelu * dY[begin + i];
                }
            });
        });
    }
    return output;
}

std::shared_ptr<Variable> Variable::silu() {
    data.assertValid("Variable::silu(x)");

    // silu(x) = x * sigmoid(x), with sigmoid computed as 1/(1 + e^-x)
    // through vec_exp (SIMD), mirroring gelu's structure.
    const size_t n = data.numel();
    const float* x = data.raw();

    Tensor result = data.getIs3D()
        ? Tensor::uninitialized(data.getBatchSize(), data.getRows(), data.getCols())
        : Tensor::uninitialized(data.getRows(), data.getCols());
    float* out = result.raw();

    parallel_for(n, 32768, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; i++) {
            out[i] = -x[i];
        }
        vec_exp(out + begin, out + begin, static_cast<int>(end - begin));
        for (size_t i = begin; i < end; i++) {
            out[i] = x[i] / (1.0f + out[i]);
        }
    });

    auto output = createOutput(std::move(result), this->requires_grad);

    if (this->requires_grad) {
        auto self_ptr = shared_from_this();
        output->addChild(self_ptr);
        output->setBackwardFn([self_ptr, output_weak = std::weak_ptr<Variable>(output)]() {
            auto output = output_weak.lock();
            if (!output || !output->hasGrad()) return;
            if (!self_ptr->requires_grad) return;
            self_ptr->ensureGrad();

            // d silu = sigmoid(x) * (1 + x * (1 - sigmoid(x))); the
            // sigmoid is recomputed - one vec_exp is cheaper than caching
            // a full activation tensor across the step.
            const size_t n = self_ptr->data.numel();
            const float* x = self_ptr->data.raw();
            const float* dY = output->grad.raw();
            float* dX = self_ptr->grad.raw();

            parallel_for(n, 32768, [&](size_t begin, size_t end) {
                const size_t len = end - begin;
                std::vector<float> sig(len);
                for (size_t i = 0; i < len; i++) {
                    sig[i] = -x[begin + i];
                }
                vec_exp(sig.data(), sig.data(), static_cast<int>(len));
                for (size_t i = 0; i < len; i++) {
                    float s = 1.0f / (1.0f + sig[i]);
                    float xi = x[begin + i];
                    dX[begin + i] += dY[begin + i] * s * (1.0f + xi * (1.0f - s));
                }
            });
        });
    }
    return output;
}

std::shared_ptr<Variable> Variable::mul(std::shared_ptr<Variable> other) {
    data.assertValid("Variable::mul(lhs)");
    other->data.assertValid("Variable::mul(rhs)");

    Tensor result = this->data.elementwise(other->data);
    bool needs_grad = this->requires_grad || other->requires_grad;
    auto output = createOutput(std::move(result), needs_grad);

    if (needs_grad) {
        auto self_ptr = shared_from_this();
        output->addChild(self_ptr);
        output->addChild(other);

        output->setBackwardFn([self_ptr, other, output_weak = std::weak_ptr<Variable>(output)]() {
            auto output = output_weak.lock();
            if (!output || !output->hasGrad()) return;

            if (self_ptr->requires_grad) {
                self_ptr->ensureGrad();
                Tensor d = output->grad.elementwise(other->data);
                self_ptr->grad.add_inplace(d);
            }
            if (other->requires_grad) {
                other->ensureGrad();
                Tensor d = output->grad.elementwise(self_ptr->data);
                other->grad.add_inplace(d);
            }
        });
    }
    return output;
}

std::shared_ptr<Variable> Variable::dropout(float dropout_rate, bool training) {
    if (!training || dropout_rate == 0.0f) {
        return shared_from_this();
    }

    data.assertValid("Variable::dropout(x)");
    float scale = 1.0f / (1.0f - dropout_rate);
    Tensor mask = data.getIs3D()
        ? Tensor::uninitialized(data.getBatchSize(), data.getRows(), data.getCols())
        : Tensor::uninitialized(data.getRows(), data.getCols());

    fill_dropout_mask(mask.raw(), mask.numel(), dropout_rate, scale);

    Tensor result = this->data.elementwise(mask);
    auto output = createOutput(std::move(result), this->requires_grad);
    
    if (this->requires_grad) {
        auto self_ptr = shared_from_this();
        output->addChild(self_ptr);
        auto mask_ptr = std::make_shared<Tensor>(std::move(mask));
        output->setBackwardFn([self_ptr, output_weak = std::weak_ptr<Variable>(output), mask_ptr]() {
            auto output = output_weak.lock();
            if (!output || !output->hasGrad()) return;
            if (self_ptr->requires_grad) {
                self_ptr->ensureGrad();
                Tensor grad_tensor = output->grad.elementwise(*mask_ptr);
                self_ptr->grad.add_inplace(grad_tensor);
            }
        });
    }
    return output;
}

std::shared_ptr<Variable> Variable::log_softmax() {
    // Rows are contiguous whether the tensor is 2D or 3D, so both cases are
    // one loop over batch*rows. The exp goes through vec_exp (SIMD).
    const size_t cols = data.getCols();
    const size_t total_rows = data.getIs3D()
        ? data.getBatchSize() * data.getRows()
        : data.getRows();

    Tensor result = data.getIs3D()
        ? Tensor::uninitialized(data.getBatchSize(), data.getRows(), data.getCols())
        : Tensor::uninitialized(data.getRows(), data.getCols());

    const float* in = data.raw();
    float* out = result.raw();

    parallel_for(total_rows, 8, [&](size_t begin, size_t end) {
        std::vector<float> exps(cols);
        for (size_t i = begin; i < end; i++) {
            const float* row_in = in + i * cols;
            float* row_out = out + i * cols;

            float max_val = row_in[0];
            for (size_t j = 1; j < cols; j++) {
                max_val = std::max(max_val, row_in[j]);
            }
            for (size_t j = 0; j < cols; j++) {
                row_out[j] = row_in[j] - max_val;
            }
            vec_exp(row_out, exps.data(), static_cast<int>(cols));
            float log_sum = std::log(vec_sum(exps.data(), static_cast<int>(cols)));
            for (size_t j = 0; j < cols; j++) {
                row_out[j] -= log_sum;
            }
        }
    });

    auto output = createOutput(std::move(result), this->requires_grad);

    if (this->requires_grad) {
        auto self_ptr = shared_from_this();
        output->addChild(self_ptr);

        output->setBackwardFn([self_ptr, total_rows, cols,
                               output_weak = std::weak_ptr<Variable>(output)]() {
            auto output = output_weak.lock();
            if (!output || !output->hasGrad()) return;
            self_ptr->ensureGrad();

            // d/dx log_softmax: dX = dY - softmax(x) * sum(dY) per row,
            // where softmax(x) = exp(this node's own output data - read in
            // place rather than captured as a copy; retirement frees it
            // only after this fn runs). Accumulates into the grad tensor
            // directly.
            const float* res = output->getData().raw();
            const float* dY = output->grad.raw();
            float* dX = self_ptr->grad.raw();

            parallel_for(total_rows, 8, [&](size_t begin, size_t end) {
                std::vector<float> soft(cols);
                for (size_t i = begin; i < end; i++) {
                    const float* row_res = res + i * cols;
                    const float* row_dY = dY + i * cols;
                    float* row_dX = dX + i * cols;

                    float sum = vec_sum(row_dY, static_cast<int>(cols));
                    vec_exp(row_res, soft.data(), static_cast<int>(cols));
                    for (size_t j = 0; j < cols; j++) {
                        row_dX[j] += row_dY[j] - soft[j] * sum;
                    }
                }
            });
        });
    }
    return output;
}

// Mean negative log-likelihood over rows of log-probabilities. 2D (rows, V)
// and 3D (batch, seq, V) inputs are the same contiguous row-major rows, so
// one loop serves both. targets holds one class index per row (any shape
// with that many elements); an index outside [0, V) contributes nothing.
std::shared_ptr<Variable> Variable::nll_loss(std::shared_ptr<Variable> targets) {
    data.assertValid("Variable::nll_loss(input)");
    targets->data.assertValid("Variable::nll_loss(targets)");

    const size_t vocab = data.getCols();
    const size_t n = data.numel() / vocab;
    if (targets->data.numel() != n) {
        throw std::invalid_argument("nll_loss: " + std::to_string(targets->data.numel()) +
                                    " targets for " + std::to_string(n) + " rows");
    }

    const float* logp = data.raw();
    const float* tgt = targets->data.raw();
    float total_loss = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const int t = static_cast<int>(tgt[i]);
        if (t >= 0 && static_cast<size_t>(t) < vocab) {
            total_loss -= logp[i * vocab + t];
        }
    }
    total_loss /= static_cast<float>(n);

    Tensor loss_tensor(1, 1);
    loss_tensor.raw()[0] = total_loss;
    auto output = createOutput(std::move(loss_tensor), this->requires_grad);

    if (this->requires_grad) {
        auto self_ptr = shared_from_this();
        output->addChild(self_ptr);
        output->addChild(targets);

        output->setBackwardFn([self_ptr, targets, n, vocab,
                               output_weak = std::weak_ptr<Variable>(output)]() {
            auto output = output_weak.lock();
            if (!output || !output->hasGrad()) return;
            self_ptr->ensureGrad();

            // dL/dlogp[i, t_i] = -upstream / n; every other entry is zero,
            // so only the target entries are touched. backward() seeds the
            // upstream gradient with 1, and a loss scaled before backward
            // scales these gradients with it.
            const float scale = -output->grad.raw()[0] / static_cast<float>(n);
            float* g = self_ptr->grad.raw();
            const float* tgt = targets->data.raw();
            for (size_t i = 0; i < n; i++) {
                const int t = static_cast<int>(tgt[i]);
                if (t >= 0 && static_cast<size_t>(t) < vocab) {
                    g[i * vocab + t] += scale;
                }
            }
        });
    }
    return output;
}

void Variable::topologicalSort(std::vector<std::shared_ptr<Variable>>& sorted, std::unordered_set<Variable*>& visited) {
    if (visited.find(this) != visited.end()) {
        return;
    }
    
    visited.insert(this);
    
    for (const auto& child : children) {
        child->topologicalSort(sorted, visited);
    }
    
    sorted.push_back(shared_from_this());
}

void Variable::backward() {
    if (!requires_grad) {
        std::cerr << "Warning: backward() called on Variable that doesn't require grad" << std::endl;
        return;
    }
    
    if (data.numel() != 1) {
        throw std::runtime_error("Variable::backward(): output must be scalar to auto-seed dOut=1. ""For non-scalars, provide an explicit upstream gradient.");
    }
    ensureGrad();
    grad.fill(1.0f);

    std::vector<std::shared_ptr<Variable>> sorted;
    std::unordered_set<Variable*> visited;
    topologicalSort(sorted, visited);

    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
        Variable* node = it->get();
        if (!node->backward_fn) continue;
        node->backward_fn();

        // Retire the node as the wave passes: reverse-topological order
        // means every consumer of this node's data has already run, and
        // its grad is only ever read by its own backward fn. Freeing here
        // caps backward's live set at the frontier instead of holding the
        // whole graph until release_graph(). Parameters and graph inputs
        // are leaves (no backward fn) and are never retired; the root
        // keeps its tensors because callers read the loss value after
        // backward. Clearing backward_fn also releases forward-pass
        // caches captured in the closure (softmax outputs, dropout masks).
        node->backward_fn = nullptr;
        node->children.clear();
        if (node != this) {
            node->data = Tensor();
            node->grad = Tensor();
        }
    }
}

void Variable::zeroGrad() {
    if (requires_grad && grad.numel() > 0) {
        grad.fill(0.0f);
    }
}

void Variable::release_graph() {
    for (auto& child : children) {
        if (child) {
            child->release_graph();
        }
    }

    children.clear();
    backward_fn = nullptr;
}