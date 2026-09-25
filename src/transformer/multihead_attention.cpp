#include "transformer/tensor.h"
#include "transformer/activations.h"
#include "transformer/multihead_attention.h"
#include <cmath>
#include <stdexcept>
#include <cstring>
#include "transformer/blas_wrapper.h"
#include "transformer/parallel.h"
#include <memory>
#include <string>
#include <vector>

namespace {

inline void fill_bias_rows(float* out, const float* bias, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        std::memcpy(out + static_cast<size_t>(i) * cols, bias, cols * sizeof(float));
    }
}

inline void add_column_sums(const float* x, int rows, int cols, float* out) {
    for (int i = 0; i < rows; i++) {
        const float* row = x + static_cast<size_t>(i) * cols;
        for (int j = 0; j < cols; j++) {
            out[j] += row[j];
        }
    }
}

// cos/sin tables for RoPE: row i holds cos/sin(i * theta_j) for the
// head_size/2 rotation frequencies theta_j = 10000^(-2j/head_size).
void build_rope_tables(int seq_len, int head_size,
                       std::vector<float>& cos_t, std::vector<float>& sin_t) {
    const int half = head_size / 2;
    cos_t.resize(static_cast<size_t>(seq_len) * half);
    sin_t.resize(static_cast<size_t>(seq_len) * half);
    for (int j = 0; j < half; j++) {
        const float theta = std::pow(10000.0f, -2.0f * j / head_size);
        for (int i = 0; i < seq_len; i++) {
            cos_t[static_cast<size_t>(i) * half + j] = std::cos(i * theta);
            sin_t[static_cast<size_t>(i) * half + j] = std::sin(i * theta);
        }
    }
}

// Rotates each head's (2j, 2j+1) pairs in a (seq_len, d_model) block by
// its row's position angle, in place. inverse applies the transpose
// rotation - backward through RoPE, since rotations are orthogonal.
void rope_apply(float* buf, int seq_len, int d_model, int num_heads, int head_size,
                const float* cos_t, const float* sin_t, bool inverse) {
    const int half = head_size / 2;
    for (int i = 0; i < seq_len; i++) {
        const float* c_row = cos_t + static_cast<size_t>(i) * half;
        const float* s_row = sin_t + static_cast<size_t>(i) * half;
        float* row = buf + static_cast<size_t>(i) * d_model;
        for (int h = 0; h < num_heads; h++) {
            float* head = row + h * head_size;
            for (int j = 0; j < half; j++) {
                const float c = c_row[j];
                const float s = inverse ? -s_row[j] : s_row[j];
                const float a = head[2 * j];
                const float b = head[2 * j + 1];
                head[2 * j] = a * c - b * s;
                head[2 * j + 1] = a * s + b * c;
            }
        }
    }
}

// Same values in a new shape (equal numel, both contiguous row-major), with
// the gradient passed straight back. Used to run 2D input through the
// batched attention path.
std::shared_ptr<Variable> relayout(const std::shared_ptr<Variable>& src, Tensor&& shaped) {
    std::memcpy(shaped.raw(), src->getData().raw(), shaped.numel() * sizeof(float));
    auto out = Variable::create(std::move(shaped), src->requiresGrad());
    if (src->requiresGrad()) {
        out->addChild(src);
        out->setBackwardFn([src, out_weak = std::weak_ptr<Variable>(out)]() {
            auto out = out_weak.lock();
            if (!out || !out->hasGrad()) return;
            src->ensureGrad();
            float* dst = src->getGrad().raw();
            blas_vadd(dst, out->getGrad().raw(), dst, out->getGrad().numel());
        });
    }
    return out;
}

}  // namespace


MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads, float dropout_rate,
                                       bool rope) :
    d_model(d_model),
    num_heads(num_heads),
    dropout_rate(dropout_rate),
    rope_(rope)
{
    if (num_heads == 0 || (d_model % num_heads) != 0) {
        throw std::invalid_argument("d_model must be divisible by num_heads and num_heads > 0");
    }
    if (rope && (d_model / num_heads) % 2 != 0) {
        throw std::invalid_argument("RoPE requires an even head size");
    }

    Tensor wq_tensor(d_model, d_model);
    Tensor wk_tensor(d_model, d_model);
    Tensor wv_tensor(d_model, d_model);
    Tensor wo_tensor(d_model, d_model);
    
    wq_tensor.xavier(d_model, d_model);
    wk_tensor.xavier(d_model, d_model);
    wv_tensor.xavier(d_model, d_model);
    wo_tensor.xavier(d_model, d_model);

    W_q = Variable::create(wq_tensor, true);
    W_k = Variable::create(wk_tensor, true);
    W_v = Variable::create(wv_tensor, true);
    W_o = Variable::create(wo_tensor, true);

    Tensor bq_tensor(1, d_model);
    Tensor bk_tensor(1, d_model);
    Tensor bv_tensor(1, d_model);
    Tensor bo_tensor(1, d_model);

    bq_tensor.fill(0.0f);
    bk_tensor.fill(0.0f);
    bv_tensor.fill(0.0f);
    bo_tensor.fill(0.0f);

    b_q = Variable::create(bq_tensor, true);
    b_k = Variable::create(bk_tensor, true);
    b_v = Variable::create(bv_tensor, true);
    b_o = Variable::create(bo_tensor, true);
}


std::shared_ptr<Variable> MultiHeadAttention::forward(std::shared_ptr<Variable> input, bool training) const {
    const Tensor& input_tensor = input->getData();

    if (input_tensor.getCols() != static_cast<size_t>(d_model)) {
        throw std::invalid_argument("MultiHeadAttention: input width " +
                                    std::to_string(input_tensor.getCols()) +
                                    " does not match d_model " + std::to_string(d_model));
    }

    if (!input_tensor.getIs3D()) {
        // A 2D (seq, d_model) input is a single sequence: run it through the
        // batched path as batch 1 and hand back a 2D result, so there is one
        // forward/backward implementation (dropout masks, RoPE and all) to
        // keep correct. Both relayouts are one memcpy of the activations;
        // training and generation never take this path.
        const size_t S = input_tensor.getRows();
        auto batched = forward(relayout(input, Tensor::uninitialized(1, S, input_tensor.getCols())),
                               training);
        return relayout(batched, Tensor::uninitialized(S, batched->getData().getCols()));
    }

    // (batch, seq, d_model): the training path.
    //
    // Layout notes: a 3D tensor is contiguous, so all four projections
    // run as single (batch*seq, d) sgemms. Attention itself is
    // batch*heads independent (seq, head_size) problems, executed in
    // parallel with per-task scratch. The softmax output (and the
    // attention-dropout mask, when active) are cached for the backward
    // pass, which both avoids recomputing them and makes the dropout
    // gradient exact instead of ignoring the mask.
    const int batch_size = input_tensor.getBatchSize();
    const int seq_len = input_tensor.getRows();
    const int head_size = d_model / num_heads;
    const int flat = batch_size * seq_len;
    const int d = d_model;
    const int H = num_heads;
    const int S = seq_len;

    const bool use_attn_dropout = training && dropout_rate > 0.0f;
    const float keep_scale = 1.0f / (1.0f - dropout_rate);

    Tensor causal_mask = Tensor::create_causal_mask(S);
    const float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_size));
    const float* mask_data = causal_mask.raw();

    // Q/K/V/concat live on the heap (shared_ptr) because the backward
    // closure needs them after forward returns.
    auto Q = std::make_shared<Tensor>(Tensor::uninitialized(flat, d));
    auto K = std::make_shared<Tensor>(Tensor::uninitialized(flat, d));
    auto V = std::make_shared<Tensor>(Tensor::uninitialized(flat, d));
    auto concat = std::make_shared<Tensor>(Tensor::uninitialized(flat, d));

    const float* in_data = input_tensor.raw();
    auto project = [&](const Tensor& W, const Tensor& b, Tensor& out) {
        fill_bias_rows(out.raw(), b.raw(), flat, d);
        blas_sgemm_ex(in_data, W.raw(), out.raw(), flat, d, d,
                      false, false, 1.0f, 1.0f);
    };
    project(W_q->getData(), b_q->getData(), *Q);
    project(W_k->getData(), b_k->getData(), *K);
    project(W_v->getData(), b_v->getData(), *V);

    // RoPE: rotate Q and K in place, so both the score matmuls below
    // and the cached copies the backward pass regathers are the
    // rotated values. V is untouched.
    auto rope_cos = std::make_shared<std::vector<float>>();
    auto rope_sin = std::make_shared<std::vector<float>>();
    if (rope_) {
        build_rope_tables(S, head_size, *rope_cos, *rope_sin);
        for (int b = 0; b < batch_size; b++) {
            float* q_block = Q->raw() + static_cast<size_t>(b) * S * d;
            float* k_block = K->raw() + static_cast<size_t>(b) * S * d;
            rope_apply(q_block, S, d, H, head_size, rope_cos->data(), rope_sin->data(), false);
            rope_apply(k_block, S, d, H, head_size, rope_cos->data(), rope_sin->data(), false);
        }
    }

    // Cached softmax outputs (and dropout masks) per (batch, head).
    auto attn_cache = std::make_shared<std::vector<float>>(
        static_cast<size_t>(batch_size) * H * S * S);
    std::shared_ptr<std::vector<float>> drop_mask;
    if (use_attn_dropout) {
        drop_mask = std::make_shared<std::vector<float>>(attn_cache->size());
    }

    const float* Q_data = Q->raw();
    const float* K_data = K->raw();
    const float* V_data = V->raw();
    float* concat_data = concat->raw();
    float* attn_data_all = attn_cache->data();
    float* mask_all = drop_mask ? drop_mask->data() : nullptr;
    // One dropout stream per (batch, head) unit, reserved before the
    // parallel loop so each unit's mask is fixed by its index rather
    // than by which thread reached the RNG first.
    const uint64_t stream_base = use_attn_dropout
        ? reserve_dropout_streams(static_cast<uint64_t>(batch_size) * H) : 0;

    // Heads are column slices of the (rows, d_model) projections; BLAS
    // takes them as strided views (lda = d_model), so no per-head
    // gather/scatter copies are needed anywhere in this loop -
    // measured at ~12% of step CPU as memmove (BENCHMARKS.md #10).
    parallel_for(static_cast<size_t>(batch_size) * H, 1,
                 [&](size_t begin, size_t end) {
        Tensor scores = Tensor::uninitialized(S, S);

        for (size_t unit = begin; unit < end; unit++) {
            const int b = static_cast<int>(unit) / H;
            const int h = static_cast<int>(unit) % H;
            const size_t batch_offset = static_cast<size_t>(b) * S * d;
            const int col = h * head_size;

            const float* q_ptr = Q_data + batch_offset + col;
            const float* k_ptr = K_data + batch_offset + col;
            const float* v_ptr = V_data + batch_offset + col;

            float* scores_data = scores.raw();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        S, S, head_size,
                        1.0f, q_ptr, d, k_ptr, d,
                        0.0f, scores_data, S);
            for (int i = 0; i < S * S; i++) {
                scores_data[i] = scores_data[i] * scale_factor + mask_data[i];
            }

            // Softmax rows straight into the cache slice (same math as
            // Tensor::softmax: subtract max, vec_exp, scale by 1/sum).
            float* attn_unit = attn_data_all + unit * S * S;
            for (int i = 0; i < S; i++) {
                const float* row_in = scores_data + static_cast<size_t>(i) * S;
                float* row_out = attn_unit + static_cast<size_t>(i) * S;
                float max_val = row_in[0];
                for (int j = 1; j < S; j++) {
                    max_val = std::max(max_val, row_in[j]);
                }
                for (int j = 0; j < S; j++) {
                    row_out[j] = row_in[j] - max_val;
                }
                vec_exp(row_out, row_out, S);
                const float inv_sum = 1.0f / vec_sum(row_out, S);
                for (int j = 0; j < S; j++) {
                    row_out[j] *= inv_sum;
                }
            }

            const float* effective = attn_unit;
            Tensor dropped(1, 1);
            if (use_attn_dropout) {
                float* m = mask_all + unit * S * S;
                fill_dropout_mask(m, static_cast<size_t>(S) * S,
                                  dropout_rate, keep_scale, stream_base + unit);
                dropped = Tensor::uninitialized(S, S);
                float* w = dropped.raw();
                for (int i = 0; i < S * S; i++) {
                    w[i] = attn_unit[i] * m[i];
                }
                effective = w;
            }

            // attended lands directly in its concat column slice.
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        S, head_size, S,
                        1.0f, effective, S, v_ptr, d,
                        0.0f, concat_data + batch_offset + col, d);
        }
    });

    // Output projection, also flat.
    Tensor out_tensor = Tensor::uninitialized(batch_size, S, d);
    fill_bias_rows(out_tensor.raw(), b_o->getData().raw(), flat, d);
    blas_sgemm_ex(concat_data, W_o->getData().raw(), out_tensor.raw(),
                  flat, d, d, false, false, 1.0f, 1.0f);

    bool needs_grad = input->requiresGrad();
    auto output = Variable::create(std::move(out_tensor), needs_grad);

    if (needs_grad) {
        auto self_input = input;
        auto self_Wq = W_q; auto self_Wk = W_k;
        auto self_Wv = W_v; auto self_Wo = W_o;
        auto self_bq = b_q; auto self_bk = b_k;
        auto self_bv = b_v; auto self_bo = b_o;

        output->addChild(input);
        output->addChild(W_q);
        output->addChild(W_k);
        output->addChild(W_v);
        output->addChild(W_o);
        output->addChild(b_q);
        output->addChild(b_k);
        output->addChild(b_v);
        output->addChild(b_o);

        const bool dropout_active = use_attn_dropout;
        const bool rope_active = rope_;
        output->setBackwardFn([self_input, self_Wq, self_Wk, self_Wv, self_Wo,
                               self_bq, self_bk, self_bv, self_bo,
                               Q, K, V, concat, attn_cache, drop_mask,
                               rope_cos, rope_sin, rope_active,
                               output_weak = std::weak_ptr<Variable>(output),
                               batch_size, S, d, H, head_size, flat,
                               scale_factor, dropout_active]() {
            auto output = output_weak.lock();
            if (!output || !output->hasGrad()) return;
            self_Wq->ensureGrad(); self_Wk->ensureGrad();
            self_Wv->ensureGrad(); self_Wo->ensureGrad();
            self_bq->ensureGrad(); self_bk->ensureGrad();
            self_bv->ensureGrad(); self_bo->ensureGrad();
            if (self_input->requiresGrad()) self_input->ensureGrad();

            const float* dOut = output->getGrad().raw();

            // Output projection gradients: single flat sgemms.
            // dWo += concat^T @ dOut ; dbo += column sums of dOut.
            blas_sgemm_ex(concat->raw(), dOut, self_Wo->getGrad().raw(),
                          d, d, flat, true, false, 1.0f, 1.0f);
            add_column_sums(dOut, flat, d, self_bo->getGrad().raw());

            // dConcat = dOut @ Wo^T
            Tensor dConcat = Tensor::uninitialized(flat, d);
            blas_sgemm_ex(dOut, self_Wo->getData().raw(), dConcat.raw(),
                          flat, d, d, false, true, 1.0f, 0.0f);

            Tensor dQ = Tensor::uninitialized(flat, d);
            Tensor dK = Tensor::uninitialized(flat, d);
            Tensor dV = Tensor::uninitialized(flat, d);
            float* dQ_data = dQ.raw();
            float* dK_data = dK.raw();
            float* dV_data = dV.raw();
            const float* dConcat_data = dConcat.raw();
            const float* Q_data = Q->raw();
            const float* K_data = K->raw();
            const float* V_data = V->raw();
            const float* attn_all = attn_cache->data();
            const float* mask_all = drop_mask ? drop_mask->data() : nullptr;

            // Strided views throughout, mirroring forward: head slices
            // read with lda = d, and dQ/dK/dV written directly into
            // their disjoint column slices (each unit owns one).
            parallel_for(static_cast<size_t>(batch_size) * H, 1,
                         [&](size_t begin, size_t end) {
                Tensor dWeights = Tensor::uninitialized(S, S);
                Tensor dScores = Tensor::uninitialized(S, S);
                Tensor effective = Tensor::uninitialized(S, S);

                for (size_t unit = begin; unit < end; unit++) {
                    const int b = static_cast<int>(unit) / H;
                    const int h = static_cast<int>(unit) % H;
                    const size_t batch_offset = static_cast<size_t>(b) * S * d;
                    const int col = h * head_size;

                    const float* q_ptr = Q_data + batch_offset + col;
                    const float* k_ptr = K_data + batch_offset + col;
                    const float* v_ptr = V_data + batch_offset + col;
                    const float* dAtt_ptr = dConcat_data + batch_offset + col;

                    const float* W = attn_all + unit * S * S;  // pre-dropout softmax
                    const float* W_eff = W;
                    if (dropout_active) {
                        const float* m = mask_all + unit * S * S;
                        float* e = effective.raw();
                        for (int i = 0; i < S * S; i++) {
                            e[i] = W[i] * m[i];
                        }
                        W_eff = e;
                    }

                    // attended = W_eff @ V_head:
                    //   dW_eff = dAttended @ V_head^T ; dV_head = W_eff^T @ dAttended
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                S, S, head_size,
                                1.0f, dAtt_ptr, d, v_ptr, d,
                                0.0f, dWeights.raw(), S);
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                S, head_size, S,
                                1.0f, W_eff, S, dAtt_ptr, d,
                                0.0f, dV_data + batch_offset + col, d);

                    float* dW = dWeights.raw();
                    if (dropout_active) {
                        // Through the dropout: dW = dW_eff * mask.
                        const float* m = mask_all + unit * S * S;
                        for (int i = 0; i < S * S; i++) {
                            dW[i] *= m[i];
                        }
                    }

                    // Softmax backward per row:
                    //   dScores = W * (dW - sum(dW * W)) * scale
                    float* dS = dScores.raw();
                    for (int i = 0; i < S; i++) {
                        const float* w_row = W + static_cast<size_t>(i) * S;
                        const float* dw_row = dW + static_cast<size_t>(i) * S;
                        float* ds_row = dS + static_cast<size_t>(i) * S;
                        float sum = 0.0f;
                        for (int j = 0; j < S; j++) {
                            sum += dw_row[j] * w_row[j];
                        }
                        for (int j = 0; j < S; j++) {
                            ds_row[j] = w_row[j] * (dw_row[j] - sum) * scale_factor;
                        }
                    }

                    // dQ_head = dScores @ K_head ; dK_head = dScores^T @ Q_head
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                S, head_size, S,
                                1.0f, dS, S, k_ptr, d,
                                0.0f, dQ_data + batch_offset + col, d);
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                S, head_size, S,
                                1.0f, dS, S, q_ptr, d,
                                0.0f, dK_data + batch_offset + col, d);
                }
            });

            // dQ/dK so far are gradients w.r.t. the ROTATED Q and K
            // (forward cached the rotated values). The inverse rotation
            // maps them back to pre-RoPE projection space before the
            // weight/bias/input gradients below.
            if (rope_active) {
                for (int b = 0; b < batch_size; b++) {
                    rope_apply(dQ_data + static_cast<size_t>(b) * S * d,
                               S, d, H, head_size,
                               rope_cos->data(), rope_sin->data(), true);
                    rope_apply(dK_data + static_cast<size_t>(b) * S * d,
                               S, d, H, head_size,
                               rope_cos->data(), rope_sin->data(), true);
                }
            }

            // Projection gradients, all flat single sgemms with beta=1
            // accumulation. dInput sums the three projection paths.
            const float* in_data = self_input->getData().raw();
            blas_sgemm_ex(in_data, dQ_data, self_Wq->getGrad().raw(),
                          d, d, flat, true, false, 1.0f, 1.0f);
            blas_sgemm_ex(in_data, dK_data, self_Wk->getGrad().raw(),
                          d, d, flat, true, false, 1.0f, 1.0f);
            blas_sgemm_ex(in_data, dV_data, self_Wv->getGrad().raw(),
                          d, d, flat, true, false, 1.0f, 1.0f);

            add_column_sums(dQ_data, flat, d, self_bq->getGrad().raw());
            add_column_sums(dK_data, flat, d, self_bk->getGrad().raw());
            add_column_sums(dV_data, flat, d, self_bv->getGrad().raw());

            if (self_input->requiresGrad()) {
                float* dIn = self_input->getGrad().raw();
                blas_sgemm_ex(dQ_data, self_Wq->getData().raw(), dIn,
                              flat, d, d, false, true, 1.0f, 1.0f);
                blas_sgemm_ex(dK_data, self_Wk->getData().raw(), dIn,
                              flat, d, d, false, true, 1.0f, 1.0f);
                blas_sgemm_ex(dV_data, self_Wv->getData().raw(), dIn,
                              flat, d, d, false, true, 1.0f, 1.0f);
            }
        });
    }

    auto final_output = output;
    if (training && dropout_rate > 0.0f) {
        final_output = output->dropout(dropout_rate, training);
    }
    return final_output;
}
