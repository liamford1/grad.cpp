#include "grad/transformer/variable.h"
#include "grad/transformer/activations.h"
#include "grad/transformer/blas_wrapper.h"
#include "grad/transformer/parallel.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace grad {

namespace {

// Sums a broadcasting add's output gradient g (viewed as batch x rows x
// cols) down to a 2D operand of shape (R, C), over every axis the operand
// was broadcast along: the batch always, rows if br, columns if bc.
Tensor reduce_broadcast(const Tensor& g, size_t R, size_t C, bool br, bool bc) {
    Tensor out(R, C);
    const size_t B = g.getBatchSize();
    const size_t GR = g.getRows();
    const size_t GC = g.getCols();
    const float* g_ptr = g.raw();
    float* out_ptr = out.raw();

    if (br && bc) {
        float s = 0.0f;
        for (size_t i = 0; i < g.numel(); ++i) s += g_ptr[i];
        out_ptr[0] = s;
    } else if (br) {
        for (size_t j = 0; j < C; ++j) {
            float s = 0.0f;
            for (size_t b = 0; b < B; ++b) {
                for (size_t ii = 0; ii < GR; ++ii) {
                    s += g_ptr[b * GR * GC + ii * GC + j];
                }
            }
            out_ptr[j] = s;
        }
    } else if (bc) {
        for (size_t i = 0; i < R; ++i) {
            float s = 0.0f;
            for (size_t b = 0; b < B; ++b) {
                const float* row = g_ptr + b * GR * GC + i * GC;
                for (size_t jj = 0; jj < GC; ++jj) s += row[jj];
            }
            out_ptr[i] = s;
        }
    } else {
        for (size_t b = 0; b < B; ++b) {
            for (size_t i = 0; i < R; ++i) {
                const float* row = g_ptr + b * GR * GC + i * GC;
                float* out_row = out_ptr + i * C;
                for (size_t j = 0; j < C; ++j) out_row[j] += row[j];
            }
        }
    }
    return out;
}

// Adds the gradient of an add's output, dO, into grad, the gradient of the
// operand whose data is x. Fast paths cover the two shapes that dominate
// training: same-shape adds (residual connections) accumulate directly,
// and row-vector biases reduce via cache-friendly row-major column sums.
// Any other 2D operand goes through the general reduction.
void accumulate_add_grad(const Tensor& dO, const Tensor& x, Tensor& grad) {
    if (x.shape() == dO.shape()) {
        blas_vadd(grad.raw(), dO.raw(), grad.raw(), dO.numel());
        return;
    }
    if (x.getIs3D()) {
        throw std::runtime_error("add backward: operand " + x.shape().to_string()
                                 + " does not broadcast to " + dO.shape().to_string());
    }
    const size_t C = dO.getCols();
    if (x.getRows() == 1 && x.getCols() == C && C > 1) {
        const size_t rows = dO.getFlatRows();
        const float* g = dO.raw();
        float* out = grad.raw();
        for (size_t i = 0; i < rows; i++) {
            const float* row = g + i * C;
            for (size_t j = 0; j < C; j++) {
                out[j] += row[j];
            }
        }
        return;
    }
    const bool br = x.getRows() == 1 && dO.getRows() > 1;
    const bool bc = x.getCols() == 1 && dO.getCols() > 1;
    grad.add_inplace(reduce_broadcast(dO, x.getRows(), x.getCols(), br, bc));
}

}  // namespace

// Grads stay empty until ensureGrad() - see the header note on lazy
// gradient allocation.
Variable::Variable(Private, const Tensor& value, bool needs_grad)
    : data(value), requires_grad(needs_grad) {}

Variable::Variable(Private, Tensor&& value, bool needs_grad)
    : data(std::move(value)), requires_grad(needs_grad) {}

Variable::Variable(Private, size_t rows, size_t cols, bool needs_grad)
    : data(rows, cols), requires_grad(needs_grad) {}

Variable::Variable(Private, size_t batch_size, size_t rows, size_t cols, bool needs_grad)
    : data(batch_size, rows, cols), requires_grad(needs_grad) {}

void Variable::ensureGrad() {
    if (!requires_grad || grad.numel() > 0) return;
    // Zero-filled, so the grad is accumulation-ready.
    grad = Tensor::zeros_like(data);
}

std::shared_ptr<Variable> Variable::create(const Tensor& data, bool requires_grad) {
    return std::make_shared<Variable>(Private{}, data, requires_grad);
}

std::shared_ptr<Variable> Variable::create(Tensor&& data, bool requires_grad) {
    return std::make_shared<Variable>(Private{}, std::move(data), requires_grad);
}

std::shared_ptr<Variable> Variable::create(size_t rows, size_t cols, bool requires_grad) {
    return std::make_shared<Variable>(Private{}, rows, cols, requires_grad);
}

std::shared_ptr<Variable> Variable::create(size_t batch_size, size_t rows, size_t cols,
                                           bool requires_grad) {
    return std::make_shared<Variable>(Private{}, batch_size, rows, cols, requires_grad);
}

std::shared_ptr<Variable> Variable::createOutput(Tensor&& result, bool needs_grad) {
    return std::make_shared<Variable>(Private{}, std::move(result), needs_grad);
}

std::shared_ptr<Variable> Variable::matmul(const std::shared_ptr<Variable>& other) {
    data.assertValid("Variable::matmul(lhs)");
    other->data.assertValid("Variable::matmul(rhs)");

    Tensor result = this->data.matmul(other->data);
    const bool needs_grad = compute_requires_grad(this, other);

    auto node = createOutput(std::move(result), needs_grad);

    if (needs_grad) {
        auto self_ptr = shared_from_this();
        node->setBackward({self_ptr, other}, [self_ptr, other](Variable& output) {
            self_ptr->data.assertValid("Variable::matmul(self.data)");
            other->data.assertValid("Variable::matmul(other.data)");

            if (!other->data.getIs3D()) {
                // X (flat, K) @ W (K, N) = Y (flat, N), where a 3D X is its
                // contiguous (batch*rows, K) view. Both gradients are single
                // sgemms accumulated in place (beta = 1); the sgemm's transA
                // sums dW over batch*rows with no temporaries.
                const Tensor& X = self_ptr->data;
                const Tensor& dY = output.grad;
                const size_t K = other->data.getRows();
                const size_t N = other->data.getCols();
                const size_t flat = X.getFlatRows();

                if (self_ptr->requires_grad) {
                    self_ptr->ensureGrad();
                    // dX += dY @ W^T
                    blas_sgemm_ex(dY.raw(), other->data.raw(), self_ptr->grad.raw(), flat, K, N,
                                  false, true, 1.0f, 1.0f);
                }
                if (other->requires_grad) {
                    other->ensureGrad();
                    // dW += X^T @ dY
                    blas_sgemm_ex(X.raw(), dY.raw(), other->grad.raw(), K, N, flat, true, false,
                                  1.0f, 1.0f);
                }
            } else {
                if (self_ptr->requires_grad) {
                    self_ptr->ensureGrad();
                    Tensor other_transposed = other->data.transpose();
                    Tensor self_grad = output.grad.matmul(other_transposed);
                    self_ptr->grad.add_inplace(self_grad);
                }
                if (other->requires_grad) {
                    other->ensureGrad();
                    Tensor self_transposed = self_ptr->data.transpose();
                    Tensor other_grad = self_transposed.matmul(output.grad);
                    other->grad.add_inplace(other_grad);
                }
            }
        });
    }
    return node;
}

std::shared_ptr<Variable> Variable::add(const std::shared_ptr<Variable>& other) {
    data.assertValid("Variable::add(lhs)");
    other->data.assertValid("Variable::add(rhs)");

    Tensor result = this->data.add(other->data);
    const bool needs_grad = compute_requires_grad(this, other);
    auto node = createOutput(std::move(result), needs_grad);

    if (needs_grad) {
        auto self_ptr = shared_from_this();
        node->setBackward({self_ptr, other}, [self_ptr, other](Variable& output) {
            if (self_ptr->requires_grad) self_ptr->ensureGrad();
            if (other->requires_grad) other->ensureGrad();
            if (self_ptr->requires_grad) {
                accumulate_add_grad(output.grad, self_ptr->data, self_ptr->grad);
            }
            if (other->requires_grad) {
                accumulate_add_grad(output.grad, other->data, other->grad);
            }
        });
    }
    return node;
}

std::shared_ptr<Variable> Variable::scale(float factor) {
    data.assertValid("Variable::scale(x)");

    Tensor result = this->data.scale(factor);
    const bool needs_grad = compute_requires_grad(this);
    auto node = createOutput(std::move(result), needs_grad);

    if (needs_grad) {
        auto self_ptr = shared_from_this();
        node->setBackward({self_ptr}, [self_ptr, factor](Variable& output) {
            if (self_ptr->requires_grad) {
                self_ptr->ensureGrad();
                Tensor scaled_grad = output.grad.scale(factor);
                self_ptr->grad.add_inplace(scaled_grad);
            }
        });
    }
    return node;
}

std::shared_ptr<Variable> Variable::softmax() {
    data.assertValid("Variable::softmax(x)");

    Tensor result = this->data.softmax();
    const bool needs_grad = compute_requires_grad(this);
    auto node = createOutput(std::move(result), needs_grad);

    if (needs_grad) {
        auto self_ptr = shared_from_this();
        // The softmax output needed by backward IS this node's data; the
        // retire-as-you-go backward frees it only after this fn has run,
        // so reading it here avoids capturing a full copy.
        node->setBackward({self_ptr}, [self_ptr](Variable& output) {
            const Tensor& y = output.getData();
            y.assertValid("Variable::softmax(y)");

            if (self_ptr->requires_grad) {
                self_ptr->ensureGrad();
                // Per row: dX = y * (dY - dot(y, dY)).
                Tensor temp_grad = Tensor::empty_like(y);
                const size_t rows = y.getFlatRows();
                const size_t cols = y.getCols();
                const float* result_data = y.raw();
                const float* grad_out_data = output.grad.raw();
                float* temp_grad_data = temp_grad.raw();

                for (size_t i = 0; i < rows; i++) {
                    const float* row_result = result_data + i * cols;
                    const float* row_grad_out = grad_out_data + i * cols;
                    float* row_grad = temp_grad_data + i * cols;

                    float dot_product = 0.0f;
                    for (size_t j = 0; j < cols; j++) {
                        dot_product += row_result[j] * row_grad_out[j];
                    }
                    for (size_t j = 0; j < cols; j++) {
                        row_grad[j] = row_result[j] * (row_grad_out[j] - dot_product);
                    }
                }
                self_ptr->grad.add_inplace(temp_grad);
            }
        });
    }
    return node;
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

    Tensor result = Tensor::empty_like(data);
    float* out = result.raw();

    parallel_for(n, 32768, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; i++) {
            out[i] = k * (x[i] + a * x[i] * x[i] * x[i]);
        }
        vec_tanh(out + begin, out + begin, end - begin);
        for (size_t i = begin; i < end; i++) {
            out[i] = 0.5f * x[i] * (1.0f + out[i]);
        }
    });

    const bool needs_grad = compute_requires_grad(this);
    auto node = createOutput(std::move(result), needs_grad);

    if (needs_grad) {
        auto self_ptr = shared_from_this();
        node->setBackward({self_ptr}, [self_ptr](Variable& output) {
            if (!self_ptr->requires_grad) return;
            self_ptr->ensureGrad();

            const float* x_data = self_ptr->data.raw();
            const float* dY = output.grad.raw();
            float* dX = self_ptr->grad.raw();

            parallel_for(self_ptr->data.numel(), 32768, [&](size_t begin, size_t end) {
                const size_t len = end - begin;
                std::vector<float> t(len);
                for (size_t i = 0; i < len; i++) {
                    float xi = x_data[begin + i];
                    t[i] = k * (xi + a * xi * xi * xi);
                }
                vec_tanh(t.data(), t.data(), len);

                for (size_t i = 0; i < len; i++) {
                    float xi = x_data[begin + i];
                    float tv = t[i];
                    float sech_sq = 1.0f - tv * tv;
                    float dgelu =
                        0.5f * (1.0f + tv + xi * sech_sq * k * (1.0f + 3.0f * a * xi * xi));
                    dX[begin + i] += dgelu * dY[begin + i];
                }
            });
        });
    }
    return node;
}

std::shared_ptr<Variable> Variable::silu() {
    data.assertValid("Variable::silu(x)");

    // silu(x) = x * sigmoid(x), with sigmoid computed as 1/(1 + e^-x)
    // through vec_exp (SIMD), mirroring gelu's structure.
    const size_t n = data.numel();
    const float* x = data.raw();

    Tensor result = Tensor::empty_like(data);
    float* out = result.raw();

    parallel_for(n, 32768, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; i++) {
            out[i] = -x[i];
        }
        vec_exp(out + begin, out + begin, end - begin);
        for (size_t i = begin; i < end; i++) {
            out[i] = x[i] / (1.0f + out[i]);
        }
    });

    const bool needs_grad = compute_requires_grad(this);
    auto node = createOutput(std::move(result), needs_grad);

    if (needs_grad) {
        auto self_ptr = shared_from_this();
        node->setBackward({self_ptr}, [self_ptr](Variable& output) {
            if (!self_ptr->requires_grad) return;
            self_ptr->ensureGrad();

            // d silu = sigmoid(x) * (1 + x * (1 - sigmoid(x))); the
            // sigmoid is recomputed - one vec_exp is cheaper than caching
            // a full activation tensor across the step.
            const float* x_data = self_ptr->data.raw();
            const float* dY = output.grad.raw();
            float* dX = self_ptr->grad.raw();

            parallel_for(self_ptr->data.numel(), 32768, [&](size_t begin, size_t end) {
                const size_t len = end - begin;
                std::vector<float> sig(len);
                for (size_t i = 0; i < len; i++) {
                    sig[i] = -x_data[begin + i];
                }
                vec_exp(sig.data(), sig.data(), len);
                for (size_t i = 0; i < len; i++) {
                    float s = 1.0f / (1.0f + sig[i]);
                    float xi = x_data[begin + i];
                    dX[begin + i] += dY[begin + i] * s * (1.0f + xi * (1.0f - s));
                }
            });
        });
    }
    return node;
}

std::shared_ptr<Variable> Variable::mul(const std::shared_ptr<Variable>& other) {
    data.assertValid("Variable::mul(lhs)");
    other->data.assertValid("Variable::mul(rhs)");

    Tensor result = this->data.elementwise(other->data);
    const bool needs_grad = compute_requires_grad(this, other);
    auto node = createOutput(std::move(result), needs_grad);

    if (needs_grad) {
        auto self_ptr = shared_from_this();
        node->setBackward({self_ptr, other}, [self_ptr, other](Variable& output) {
            if (self_ptr->requires_grad) {
                self_ptr->ensureGrad();
                Tensor d = output.grad.elementwise(other->data);
                self_ptr->grad.add_inplace(d);
            }
            if (other->requires_grad) {
                other->ensureGrad();
                Tensor d = output.grad.elementwise(self_ptr->data);
                other->grad.add_inplace(d);
            }
        });
    }
    return node;
}

std::shared_ptr<Variable> Variable::dropout(float dropout_rate, bool training) {
    if (!training || dropout_rate == 0.0f) {
        return shared_from_this();
    }

    data.assertValid("Variable::dropout(x)");
    float scale = 1.0f / (1.0f - dropout_rate);
    Tensor mask = Tensor::empty_like(data);

    fill_dropout_mask(mask.raw(), mask.numel(), dropout_rate, scale);

    Tensor result = this->data.elementwise(mask);
    const bool needs_grad = compute_requires_grad(this);
    auto node = createOutput(std::move(result), needs_grad);

    if (needs_grad) {
        auto self_ptr = shared_from_this();
        auto mask_ptr = std::make_shared<Tensor>(std::move(mask));
        node->setBackward({self_ptr}, [self_ptr, mask_ptr](Variable& output) {
            if (self_ptr->requires_grad) {
                self_ptr->ensureGrad();
                Tensor grad_tensor = output.grad.elementwise(*mask_ptr);
                self_ptr->grad.add_inplace(grad_tensor);
            }
        });
    }
    return node;
}

std::shared_ptr<Variable> Variable::log_softmax() {
    // Rows are contiguous at any rank, so every batch is one loop over the
    // flat rows. The exp goes through vec_exp (SIMD).
    const size_t cols = data.getCols();
    const size_t total_rows = data.getFlatRows();

    Tensor result = Tensor::empty_like(data);

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
            vec_exp(row_out, exps.data(), cols);
            float log_sum = std::log(vec_sum(exps.data(), cols));
            for (size_t j = 0; j < cols; j++) {
                row_out[j] -= log_sum;
            }
        }
    });

    const bool needs_grad = compute_requires_grad(this);
    auto node = createOutput(std::move(result), needs_grad);

    if (needs_grad) {
        auto self_ptr = shared_from_this();
        node->setBackward({self_ptr}, [self_ptr, total_rows, cols](Variable& output) {
            self_ptr->ensureGrad();

            // d/dx log_softmax: dX = dY - softmax(x) * sum(dY) per row,
            // where softmax(x) = exp(this node's own output data - read in
            // place rather than captured as a copy; retirement frees it
            // only after this fn runs). Accumulates into the grad tensor
            // directly.
            const float* res = output.getData().raw();
            const float* dY = output.grad.raw();
            float* dX = self_ptr->grad.raw();

            parallel_for(total_rows, 8, [&](size_t begin, size_t end) {
                std::vector<float> soft(cols);
                for (size_t i = begin; i < end; i++) {
                    const float* row_res = res + i * cols;
                    const float* row_dY = dY + i * cols;
                    float* row_dX = dX + i * cols;

                    float sum = vec_sum(row_dY, cols);
                    vec_exp(row_res, soft.data(), cols);
                    for (size_t j = 0; j < cols; j++) {
                        row_dX[j] += row_dY[j] - soft[j] * sum;
                    }
                }
            });
        });
    }
    return node;
}

// Mean negative log-likelihood over rows of log-probabilities. 2D (rows, V)
// and 3D (batch, seq, V) inputs are the same contiguous row-major rows, so
// one loop serves both. targets holds one class index per row (any shape
// with that many elements); an index outside [0, V) contributes nothing.
std::shared_ptr<Variable> Variable::nll_loss(const std::shared_ptr<Variable>& targets) {
    data.assertValid("Variable::nll_loss(input)");
    targets->data.assertValid("Variable::nll_loss(targets)");

    const size_t vocab = data.getCols();
    const size_t n = data.numel() / vocab;
    if (targets->data.numel() != n) {
        throw std::invalid_argument("nll_loss: " + std::to_string(targets->data.numel())
                                    + " targets for " + std::to_string(n) + " rows");
    }

    const float* logp = data.raw();
    const float* tgt = targets->data.raw();
    float total_loss = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const int t = static_cast<int>(tgt[i]);
        if (t >= 0 && static_cast<size_t>(t) < vocab) {
            total_loss -= logp[i * vocab + static_cast<size_t>(t)];
        }
    }
    total_loss /= static_cast<float>(n);

    Tensor loss_tensor(1, 1);
    loss_tensor.raw()[0] = total_loss;
    const bool needs_grad = compute_requires_grad(this);
    auto node = createOutput(std::move(loss_tensor), needs_grad);

    if (needs_grad) {
        auto self_ptr = shared_from_this();
        node->setBackward({self_ptr, targets}, [self_ptr, targets, n, vocab](Variable& output) {
            self_ptr->ensureGrad();

            // dL/dlogp[i, t_i] = -upstream / n; every other entry is zero,
            // so only the target entries are touched. backward() seeds the
            // upstream gradient with 1, and a loss scaled before backward
            // scales these gradients with it.
            const float scale = -output.grad.raw()[0] / static_cast<float>(n);
            float* g = self_ptr->grad.raw();
            const float* target_ids = targets->data.raw();
            for (size_t i = 0; i < n; i++) {
                const int t = static_cast<int>(target_ids[i]);
                if (t >= 0 && static_cast<size_t>(t) < vocab) {
                    g[i * vocab + static_cast<size_t>(t)] += scale;
                }
            }
        });
    }
    return node;
}

void Variable::topologicalSort(std::vector<std::shared_ptr<Variable>>& sorted,
                               std::unordered_set<Variable*>& visited) {
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
    // Both are caller bugs: with no grad-requiring input there is no graph
    // to walk, and the root's gradient is seeded with 1, which only means
    // d(root)/d(root) for a scalar.
    if (!requires_grad) {
        throw std::logic_error("Variable::backward(): root does not require grad");
    }
    if (data.numel() != 1) {
        throw std::logic_error("Variable::backward(): root must be a scalar, got "
                               + std::to_string(data.numel()) + " elements");
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

}  // namespace grad
