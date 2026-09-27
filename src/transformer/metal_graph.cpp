#include "metal_graph.h"

#include "grad/transformer/activations.h"
#include "grad/transformer/metal_ops.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace grad::metal_graph {

namespace ops = metal::ops;

namespace {

void require_same_shape(const Tensor& a, const Tensor& b, const char* op) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument(std::string(op) + ": shapes " + a.shape().to_string() + " and "
                                    + b.shape().to_string() + " differ");
    }
}

ops::Broadcast broadcast_strides(const Tensor& t, size_t rows, size_t cols) {
    return {t.getIs3D() ? t.getRows() * t.getCols() : 0, t.getRows() == rows ? t.getCols() : 0,
            t.getCols() == cols ? size_t{1} : size_t{0}};
}

// The add backward of variable.cpp (accumulate_add_grad): same-shape
// operands accumulate, a row-vector bias takes the column sums, any other
// 2D operand reduces over the axes it was broadcast along.
void accumulate_add_grad(const Tensor& dO, const Tensor& x, Variable& target) {
    float* g = grad_for_write(target);
    if (x.shape() == dO.shape()) {
        ops::accumulate(g, dO.device_data(), dO.numel());
        return;
    }
    if (x.getIs3D()) {
        throw std::runtime_error("add backward: operand " + x.shape().to_string()
                                 + " does not broadcast to " + dO.shape().to_string());
    }
    const size_t C = dO.getCols();
    if (x.getRows() == 1 && x.getCols() == C && C > 1) {
        ops::column_sums(dO.device_data(), dO.getFlatRows(), C, g);
        return;
    }
    const bool br = x.getRows() == 1 && dO.getRows() > 1;
    const bool bc = x.getCols() == 1 && dO.getCols() > 1;
    ops::broadcast_reduce(dO.device_data(), dO.getBatchSize(), dO.getRows(), dO.getCols(), g,
                          x.getRows(), x.getCols(), br, bc);
}

// A unary elementwise op y = f(x) with dx += f'(x) * dy, where the backward
// reads the input (GELU, SiLU).
template <typename Forward, typename Backward>
VarPtr unary(const VarPtr& a, Forward forward, Backward backward) {
    const Tensor& x = a->getData();
    x.assertValid("Variable (metal) unary op");
    Tensor out = Tensor::empty_like(x);
    forward(x.device_data(), out.device_data(), x.numel());
    const bool needs_grad = compute_requires_grad(a);
    auto node = Variable::create(std::move(out), needs_grad);
    if (needs_grad) {
        node->setBackward({a}, [a, backward](Variable& output) {
            if (!a->requiresGrad()) return;
            const Tensor& in = a->getData();
            backward(in.device_data(), output.getGrad().device_data(), grad_for_write(*a),
                     in.numel());
        });
    }
    return node;
}

}  // namespace

VarPtr matmul(const VarPtr& a, const VarPtr& b) {
    const Tensor& A = a->getData();
    const Tensor& B = b->getData();
    A.assertValid("Variable::matmul(lhs)");
    B.assertValid("Variable::matmul(rhs)");
    const bool batched = B.getIs3D();
    bool compatible = A.getCols() == B.getRows();
    if (batched) {
        compatible = compatible && A.rank() == B.rank();
        for (size_t axis = 0; compatible && axis + 2 < A.rank(); axis++) {
            compatible = A.shape()[axis] == B.shape()[axis];
        }
    }
    if (A.rank() < 2 || B.rank() < 2 || !compatible) {
        throw std::invalid_argument("matmul: cannot multiply " + A.shape().to_string() + " by "
                                    + B.shape().to_string());
    }
    const size_t M = A.getRows();
    const size_t K = A.getCols();
    const size_t N = B.getCols();
    const size_t flat = A.getFlatRows();
    Tensor out = Tensor::uninitialized(A.shape().with_last_dim(N));

    // A batch of (M, K) @ (K, N), for the 3D-by-3D case.
    const auto batched_gemm = [](const float* x, const float* y, float* z, size_t m, size_t n,
                                 size_t k, bool tx, bool ty, float beta, size_t batch,
                                 size_t x_stride, size_t y_stride, size_t z_stride) {
        ops::BatchedGemm g;
        g.A = x, g.B = y, g.C = z;
        g.M = m, g.N = n, g.K = k;
        g.transA = tx, g.transB = ty, g.beta = beta;
        g.lda = tx ? m : k, g.ldb = ty ? k : n, g.ldc = n;
        g.batch = batch;
        g.a_outer = x_stride, g.b_outer = y_stride, g.c_outer = z_stride;
        ops::gemm_batched(g);
    };

    if (!batched) {
        // A 3D X is its contiguous (batch*rows, K) view: one GEMM.
        ops::gemm(A.device_data(), B.device_data(), out.device_data(), flat, N, K, false, false,
                  1.0f, 0.0f);
    } else {
        batched_gemm(A.device_data(), B.device_data(), out.device_data(), M, N, K, false, false,
                     0.0f, A.getBatchSize(), M * K, K * N, M * N);
    }

    const bool needs_grad = compute_requires_grad(a, b);
    auto node = Variable::create(std::move(out), needs_grad);
    if (needs_grad) {
        node->setBackward({a, b}, [a, b, batched, batched_gemm, M, N, K, flat](Variable& output) {
            const Tensor& X = a->getData();
            const Tensor& W = b->getData();
            const float* dY = output.getGrad().device_data();
            if (!batched) {
                // dX += dY @ W^T ; dW += X^T @ dY, both accumulated in place.
                if (a->requiresGrad()) {
                    ops::gemm(dY, W.device_data(), grad_for_write(*a), flat, K, N, false, true,
                              1.0f, 1.0f);
                }
                if (b->requiresGrad()) {
                    ops::gemm(X.device_data(), dY, grad_for_write(*b), K, N, flat, true, false,
                              1.0f, 1.0f);
                }
                return;
            }
            const size_t batch = X.getBatchSize();
            if (a->requiresGrad()) {
                batched_gemm(dY, W.device_data(), grad_for_write(*a), M, K, N, false, true, 1.0f,
                             batch, M * N, K * N, M * K);
            }
            if (b->requiresGrad()) {
                batched_gemm(X.device_data(), dY, grad_for_write(*b), K, N, M, true, false, 1.0f,
                             batch, M * K, M * N, K * N);
            }
        });
    }
    return node;
}

VarPtr add(const VarPtr& a, const VarPtr& b) {
    const Tensor& A = a->getData();
    const Tensor& B = b->getData();
    A.assertValid("Variable::add(lhs)");
    B.assertValid("Variable::add(rhs)");

    Tensor out;
    if (A.shape() == B.shape()) {
        out = Tensor::empty_like(A);
        ops::add(A.device_data(), B.device_data(), out.device_data(), A.numel());
    } else {
        // Tensor::add's broadcasting rules: a higher-rank operand fixes the
        // result, two 2D operands may both broadcast.
        const bool fixed_lhs = A.getIs3D();
        const bool fixed_rhs = B.getIs3D();
        const Shape out_shape = fixed_lhs   ? A.shape()
                                : fixed_rhs ? B.shape()
                                            : Shape{std::max(A.getRows(), B.getRows()),
                                                    std::max(A.getCols(), B.getCols())};
        const size_t R = out_shape[out_shape.rank() - 2];
        const size_t C = out_shape[out_shape.rank() - 1];
        const auto fits = [&](const Tensor& t, bool fixed) {
            if (fixed) return t.shape() == out_shape;
            return t.rank() == 2 && (t.getRows() == R || t.getRows() == 1)
                   && (t.getCols() == C || t.getCols() == 1);
        };
        if (!fits(A, fixed_lhs) || !fits(B, fixed_rhs)) {
            throw std::invalid_argument("add: shapes " + A.shape().to_string() + " and "
                                        + B.shape().to_string() + " do not broadcast");
        }
        out = Tensor::uninitialized(out_shape);
        const auto is_bias = [&](const Tensor& t) {
            return t.rank() == 2 && t.getRows() == 1 && t.getCols() == C;
        };
        if (A.shape() == out_shape && is_bias(B)) {
            ops::add_rows(A.device_data(), B.device_data(), out.device_data(), out.numel(), C, 1);
        } else if (B.shape() == out_shape && is_bias(A)) {
            ops::add_rows(B.device_data(), A.device_data(), out.device_data(), out.numel(), C, 1);
        } else {
            ops::broadcast_add(A.device_data(), broadcast_strides(A, R, C), B.device_data(),
                               broadcast_strides(B, R, C), out.device_data(), out.getBatchSize(), R,
                               C);
        }
    }

    const bool needs_grad = compute_requires_grad(a, b);
    auto node = Variable::create(std::move(out), needs_grad);
    if (needs_grad) {
        node->setBackward({a, b}, [a, b](Variable& output) {
            if (a->requiresGrad()) accumulate_add_grad(output.getGrad(), a->getData(), *a);
            if (b->requiresGrad()) accumulate_add_grad(output.getGrad(), b->getData(), *b);
        });
    }
    return node;
}

VarPtr scale(const VarPtr& a, float factor) {
    const Tensor& x = a->getData();
    x.assertValid("Variable::scale(x)");
    Tensor out = Tensor::empty_like(x);
    ops::scale(x.device_data(), factor, out.device_data(), x.numel());
    const bool needs_grad = compute_requires_grad(a);
    auto node = Variable::create(std::move(out), needs_grad);
    if (needs_grad) {
        node->setBackward({a}, [a, factor](Variable& output) {
            if (!a->requiresGrad()) return;
            const Tensor& dY = output.getGrad();
            ops::accumulate_scaled(grad_for_write(*a), dY.device_data(), factor, dY.numel());
        });
    }
    return node;
}

VarPtr softmax(const VarPtr& a) {
    const Tensor& x = a->getData();
    x.assertValid("Variable::softmax(x)");
    Tensor out = Tensor::empty_like(x);
    ops::softmax(x.device_data(), out.device_data(), x.getFlatRows(), x.getCols());
    const bool needs_grad = compute_requires_grad(a);
    auto node = Variable::create(std::move(out), needs_grad);
    if (needs_grad) {
        node->setBackward({a}, [a](Variable& output) {
            if (!a->requiresGrad()) return;
            const Tensor& y = output.getData();
            ops::softmax_backward(y.device_data(), output.getGrad().device_data(),
                                  grad_for_write(*a), y.getFlatRows(), y.getCols());
        });
    }
    return node;
}

VarPtr gelu(const VarPtr& a) {
    return unary(a, ops::gelu, ops::gelu_backward);
}

VarPtr silu(const VarPtr& a) {
    return unary(a, ops::silu, ops::silu_backward);
}

VarPtr mul(const VarPtr& a, const VarPtr& b) {
    const Tensor& A = a->getData();
    const Tensor& B = b->getData();
    A.assertValid("Variable::mul(lhs)");
    B.assertValid("Variable::mul(rhs)");
    require_same_shape(A, B, "elementwise");
    Tensor out = Tensor::empty_like(A);
    ops::mul(A.device_data(), B.device_data(), out.device_data(), A.numel());
    const bool needs_grad = compute_requires_grad(a, b);
    auto node = Variable::create(std::move(out), needs_grad);
    if (needs_grad) {
        node->setBackward({a, b}, [a, b](Variable& output) {
            const Tensor& dY = output.getGrad();
            if (a->requiresGrad()) {
                ops::mul_accumulate(grad_for_write(*a), dY.device_data(),
                                    b->getData().device_data(), dY.numel());
            }
            if (b->requiresGrad()) {
                ops::mul_accumulate(grad_for_write(*b), dY.device_data(),
                                    a->getData().device_data(), dY.numel());
            }
        });
    }
    return node;
}

VarPtr dropout(const VarPtr& a, float rate) {
    const Tensor& x = a->getData();
    x.assertValid("Variable::dropout(x)");
    const float scale = 1.0f / (1.0f - rate);
    auto mask = std::make_shared<Tensor>(Tensor::empty_like(x));
    Tensor out = Tensor::empty_like(x);
    // One stream per call, taken from the same counter in the same order as
    // the CPU path, so the mask is the CPU's mask.
    ops::dropout(x.device_data(), mask->device_data(), out.device_data(), x.numel(), 1,
                 current_dropout_seed(), reserve_dropout_streams(1), rate, scale);
    const bool needs_grad = compute_requires_grad(a);
    auto node = Variable::create(std::move(out), needs_grad);
    if (needs_grad) {
        node->setBackward({a}, [a, mask](Variable& output) {
            if (!a->requiresGrad()) return;
            const Tensor& dY = output.getGrad();
            ops::mul_accumulate(grad_for_write(*a), dY.device_data(), mask->device_data(),
                                dY.numel());
        });
    }
    return node;
}

VarPtr log_softmax(const VarPtr& a) {
    const Tensor& x = a->getData();
    x.assertValid("Variable::log_softmax(x)");
    Tensor out = Tensor::empty_like(x);
    ops::log_softmax(x.device_data(), out.device_data(), x.getFlatRows(), x.getCols());
    const bool needs_grad = compute_requires_grad(a);
    auto node = Variable::create(std::move(out), needs_grad);
    if (needs_grad) {
        node->setBackward({a}, [a](Variable& output) {
            if (!a->requiresGrad()) return;
            const Tensor& y = output.getData();
            ops::log_softmax_backward(y.device_data(), output.getGrad().device_data(),
                                      grad_for_write(*a), y.getFlatRows(), y.getCols());
        });
    }
    return node;
}

VarPtr nll_loss(const VarPtr& a, const VarPtr& targets) {
    const Tensor& logp = a->getData();
    logp.assertValid("Variable::nll_loss(input)");
    targets->getData().assertValid("Variable::nll_loss(targets)");
    const size_t vocab = logp.getCols();
    const size_t n = logp.numel() / vocab;
    if (targets->getData().numel() != n) {
        throw std::invalid_argument("nll_loss: " + std::to_string(targets->getData().numel())
                                    + " targets for " + std::to_string(n) + " rows");
    }
    Tensor loss = Tensor::uninitialized(1, 1);
    ops::nll_loss(logp.device_data(), targets->getData().device_data(), loss.device_data(), n,
                  vocab);
    const bool needs_grad = compute_requires_grad(a);
    auto node = Variable::create(std::move(loss), needs_grad);
    if (needs_grad) {
        node->setBackward({a, targets}, [a, targets, n, vocab](Variable& output) {
            if (!a->requiresGrad()) return;
            ops::nll_loss_backward(output.getGrad().device_data(), targets->getData().device_data(),
                                   grad_for_write(*a), n, vocab);
        });
    }
    return node;
}

VarPtr cross_entropy(const VarPtr& logits, const VarPtr& targets) {
    const Tensor& x = logits->getData();
    x.assertValid("Variable::cross_entropy(logits)");
    targets->getData().assertValid("Variable::cross_entropy(targets)");
    const size_t vocab = x.getCols();
    const size_t n = x.numel() / vocab;
    if (targets->getData().numel() != n) {
        throw std::invalid_argument("cross_entropy: " + std::to_string(targets->getData().numel())
                                    + " targets for " + std::to_string(n) + " rows");
    }
    // Per row: max and log-sum-exp, all the backward pass needs besides the
    // logits themselves.
    auto stats = std::make_shared<Tensor>(Tensor::uninitialized(n, 2));
    Tensor row_loss = Tensor::uninitialized(Shape{n});
    Tensor loss = Tensor::uninitialized(1, 1);
    ops::cross_entropy(x.device_data(), targets->getData().device_data(), stats->device_data(),
                       row_loss.device_data(), loss.device_data(), n, vocab);
    const bool needs_grad = compute_requires_grad(logits);
    auto node = Variable::create(std::move(loss), needs_grad);
    if (needs_grad) {
        node->setBackward({logits, targets}, [logits, targets, stats, n, vocab](Variable& output) {
            if (!logits->requiresGrad()) return;
            ops::cross_entropy_backward(logits->getData().device_data(), stats->device_data(),
                                        targets->getData().device_data(),
                                        output.getGrad().device_data(), grad_for_write(*logits), n,
                                        vocab);
        });
    }
    return node;
}

}  // namespace grad::metal_graph
