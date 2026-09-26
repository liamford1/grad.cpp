#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

// GPU kernels of the resident stream (docs/design/metal-resident.md). Each
// call encodes work and returns without waiting for it; the stream runs
// calls in the order they were made, so a kernel sees every earlier
// kernel's writes. Results become visible to the CPU through the fenced
// Tensor accessors.
//
// Every pointer must come from Tensor::device_data() of a tensor allocated
// in Metal mode, or point inside one (a head's column slice, say); anything
// else throws std::logic_error before encoding. Extents are element counts.
// "Accumulates" means the kernel adds into its output, as the CPU backward
// passes do. Reductions run in a fixed order, so repeated calls give
// bitwise identical results; they are not bitwise equal to the CPU's
// orders, and the tests bound the difference.
//
// Without the Metal backend these are stubs that throw std::logic_error;
// they are unreachable, since metal_mode() cannot become true there.
namespace grad::metal::ops {

// ---- elementwise
void fill(float* y, size_t n, float value);                               // y = value
void copy(const float* x, float* y, size_t n);                            // y = x
void add(const float* a, const float* b, float* y, size_t n);             // y = a + b
void accumulate(float* y, const float* x, size_t n);                      // y += x
void accumulate_scaled(float* y, const float* x, float s, size_t n);      // y += x * s
void mul(const float* a, const float* b, float* y, size_t n);             // y = a * b
void mul_accumulate(float* y, const float* a, const float* b, size_t n);  // y += a * b
void scale(const float* x, float s, float* y, size_t n);                  // y = x * s
// y *= s[0], with s a scalar the GPU computed (the clip coefficient).
void scale_by(float* y, const float* s, size_t n);

// y[r, c] = x[r, c] + rows[r % period, c] over the (n / cols, cols) matrix:
// a bias row (period 1), or a (period, cols) position table added to every
// sequence of length period. x may alias y.
void add_rows(const float* x, const float* rows, float* y, size_t n, size_t cols, size_t period);

// Strides of an operand read as (batch, rows, cols); 0 along an axis it is
// broadcast over.
struct Broadcast {
    size_t batch = 0, row = 0, col = 0;
};
// y(batch, rows, cols) = a + b, Tensor::add's general broadcasting case.
void broadcast_add(const float* a, Broadcast sa, const float* b, Broadcast sb, float* y,
                   size_t batch, size_t rows, size_t cols);
// Accumulates into dst (dst_rows, dst_cols) the sum of g (batch, rows, cols)
// over the batch and over rows (br) and columns (bc) dst was broadcast along.
void broadcast_reduce(const float* g, size_t batch, size_t rows, size_t cols, float* dst,
                      size_t dst_rows, size_t dst_cols, bool br, bool bc);
// Accumulates the column sums of x (rows, cols) into dst (cols): the bias
// gradient. With x viewed as (batch, S*d), the positional-table gradient.
void column_sums(const float* x, size_t rows, size_t cols, float* dst);

// ---- activations (the CPU's tanh-approximated GELU; SiLU = x sigmoid(x))
void gelu(const float* x, float* y, size_t n);
void gelu_backward(const float* x, const float* dy, float* dx, size_t n);  // accumulates
void silu(const float* x, float* y, size_t n);
void silu_backward(const float* x, const float* dy, float* dx, size_t n);  // accumulates

// ---- norms over the last dimension of (rows, d)
// LayerNorm, or RMSNorm (no mean, no beta) with rms; mean and rstd (rows)
// receive what the backward pass needs.
void layer_norm(const float* x, const float* gamma, const float* beta, float* y, float* mean,
                float* rstd, size_t rows, size_t d, float eps, bool rms);
// Accumulates into whichever of dx (rows, d), dgamma and dbeta (d) is
// non-null (dbeta is ignored with rms).
void layer_norm_backward(const float* x, const float* gamma, const float* dy, const float* mean,
                         const float* rstd, float* dx, float* dgamma, float* dbeta, size_t rows,
                         size_t d, bool rms);

// ---- softmax family over rows of (rows, cols)
void softmax(const float* x, float* y, size_t rows, size_t cols);
void softmax_backward(const float* y, const float* dy, float* dx, size_t rows,
                      size_t cols);  // accumulates
void log_softmax(const float* x, float* y, size_t rows, size_t cols);
void log_softmax_backward(const float* y, const float* dy, float* dx, size_t rows,
                          size_t cols);  // accumulates
// Causal attention softmax in place over units of (S, S) scores
// (rows = units * S): row i keeps columns j <= i, scaled by scale; masked
// entries become 0.
void attention_softmax(float* scores, size_t rows, size_t S, float scale);
// In place over dp: dp' = dp * mask (mask may be null: no dropout), then
// dS = P * (dp' - rowsum(dp' * P)) * scale.
void attention_softmax_backward(const float* P, float* dp, const float* mask, size_t rows, size_t S,
                                float scale);

// ---- losses. Targets are class indices stored as floats; an index outside
// [0, vocab) contributes nothing.
// loss[0] = -mean over rows of logp[r, t_r]
void nll_loss(const float* logp, const float* targets, float* loss, size_t rows, size_t vocab);
// dlogp[r, t_r] += -g[0] / rows
void nll_loss_backward(const float* g, const float* targets, float* dlogp, size_t rows,
                       size_t vocab);
// Fused log-softmax + NLL: loss[0] as nll_loss(log_softmax(logits)), and
// stats (2 * rows) keeps each row's max and log-sum-exp for the backward.
// row_loss (rows) is scratch.
void cross_entropy(const float* logits, const float* targets, float* stats, float* row_loss,
                   float* loss, size_t rows, size_t vocab);
// Accumulates into dlogits the gradient of cross_entropy's loss scaled by
// g[0].
void cross_entropy_backward(const float* logits, const float* stats, const float* targets,
                            const float* g, float* dlogits, size_t rows, size_t vocab);

// ---- embeddings and positions
// out[t, :] = table[ids[t], :] * scale; ids are validated by the caller.
void embedding(const float* ids, const float* table, float* out, size_t tokens, size_t d,
               float scale);
// Accumulates dout's rows into dtable, segment by segment: distinct token
// tokens[u] receives rows order[starts[u] .. starts[u+1]) in that order.
// The three index arrays hold uint32 values in float-typed tensor storage.
void embedding_backward(const float* tokens, const float* starts, const float* order, size_t unique,
                        const float* dout, float* dtable, size_t d, float scale);
// Rotates each head's (2j, 2j+1) pairs of x (rows, d) in place by
// position (row % S) angles, cos/sin tables (S, head / 2); inverse rotates
// back (the backward pass).
void rope(float* x, size_t rows, size_t S, size_t d, size_t head, const float* cos_t,
          const float* sin_t, bool inverse);

// ---- dropout: masks bitwise identical to fill_dropout_mask
// (activations.h). units masks of n elements each, unit u from stream
// first_stream + u under seed; mask = 0 or scale. With x non-null also
// y = x * mask.
void dropout(const float* x, float* mask, float* y, size_t n, size_t units, uint64_t seed,
             uint64_t first_stream, float rate, float scale);

// ---- optimizer
struct AdamW {
    float lr = 0.0f;
    float beta1 = 0.0f, beta2 = 0.0f;
    float inv_bc1 = 0.0f, inv_bc2 = 0.0f;  // 1 / (1 - beta^t)
    float eps = 0.0f;
    float weight_decay = 0.0f;
};
void adamw(float* w, const float* g, float* m, float* v, size_t n, const AdamW& p);

// One gradient tensor for grad_norm.
struct Span {
    const float* data = nullptr;
    size_t n = 0;
};
// out[0] = the L2 norm over every tensor; out[1] = the clip coefficient
// for max_norm (max_norm / (norm + 1e-6) above it, else 1).
void grad_norm(std::span<const Span> grads, float max_norm, float* out);

// ---- GEMM
// C = alpha * op(A) @ op(B) + beta * C over whole row-major matrices (A is
// (M, K) or, transposed, (K, M); likewise B), the contract of
// blas_sgemm_ex. Runs on MPS. With beta = 0, C is never read.
void gemm(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, bool transA,
          bool transB, float alpha, float beta);

// A batch of GEMMs over strided views: matrix z of the batch starts at
// base + (z / inner) * outer_stride + (z % inner) * inner_stride for each of
// A, B and C, and rows are ld elements apart. Attention's per-(batch, head)
// products are this with z = b * H + h: head h of a (B*S, d) projection is
// a column slice with ld = d, outer stride S*d and inner stride head_size,
// so no head is ever copied.
struct BatchedGemm {
    const float* A = nullptr;
    const float* B = nullptr;
    float* C = nullptr;
    size_t M = 0, N = 0, K = 0;
    bool transA = false, transB = false;
    float alpha = 1.0f, beta = 0.0f;
    size_t lda = 0, ldb = 0, ldc = 0;
    size_t batch = 1;
    size_t inner = 1;
    size_t a_outer = 0, a_inner = 0;
    size_t b_outer = 0, b_inner = 0;
    size_t c_outer = 0, c_inner = 0;
};
void gemm_batched(const BatchedGemm& g);

}  // namespace grad::metal::ops
