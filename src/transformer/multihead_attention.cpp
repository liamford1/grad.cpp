#include "transformer/tensor.h"
#include "transformer/activations.h"
#include "transformer/multihead_attention.h"
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cstring>
#include "transformer/blas_wrapper.h"
#include "transformer/parallel.h"
#include <memory>
#include <vector>

namespace {

// Copy a (rows, head_size) head slice out of / back into the interleaved
// (rows, d_model) layout.
inline void gather_head(const float* src, float* dst, int rows, int d_model, int head_size) {
    for (int i = 0; i < rows; i++) {
        std::memcpy(dst + static_cast<size_t>(i) * head_size,
                    src + static_cast<size_t>(i) * d_model,
                    head_size * sizeof(float));
    }
}

inline void scatter_head(const float* src, float* dst, int rows, int d_model, int head_size) {
    for (int i = 0; i < rows; i++) {
        std::memcpy(dst + static_cast<size_t>(i) * d_model,
                    src + static_cast<size_t>(i) * head_size,
                    head_size * sizeof(float));
    }
}

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

}  // namespace


MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads, float dropout_rate) : 
    d_model(d_model),
    num_heads(num_heads),
    dropout_rate(dropout_rate)
{
    if (num_heads == 0 || (d_model % num_heads) != 0) {
        throw std::invalid_argument("d_model must be divisible by num_heads and num_heads > 0");
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

MultiHeadAttention::~MultiHeadAttention() {}

std::shared_ptr<Variable> MultiHeadAttention::forward(std::shared_ptr<Variable> input, bool training) const {
    const Tensor& input_tensor = input->getData();

    if (!input_tensor.getIs3D()) {
        // 2D case
        int seq_len = input_tensor.getRows();
        int head_size = d_model / num_heads;

        auto Q = input->matmul(W_q)->add(b_q);
        auto K = input->matmul(W_k)->add(b_k);
        auto V = input->matmul(W_v)->add(b_v);

        Tensor result(seq_len, d_model);
        result.fill(0.0f);

        auto self_input = input;
        auto self_Q = Q;
        auto self_K = K;
        auto self_V = V;
        auto self_Wq = W_q;
        auto self_Wk = W_k;
        auto self_Wv = W_v;
        auto self_Wo = W_o;
        int self_num_heads = num_heads;
        int self_d_model = d_model;

        const float* Q_data = Q->getData().raw();
        const float* K_data = K->getData().raw();
        const float* V_data = V->getData().raw();
        float* result_data = result.raw();

        Tensor causal_mask = Tensor::create_causal_mask(seq_len);
        const float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_size));
        const float* mask_data = causal_mask.raw();

        Tensor Q_head(seq_len, head_size);
        Tensor K_head(seq_len, head_size);
        Tensor V_head(seq_len, head_size);
        Tensor scores(seq_len, seq_len);

        float* Q_head_data = Q_head.raw();
        float* K_head_data = K_head.raw();
        float* V_head_data = V_head.raw();
        float* scores_data = scores.raw();

        for (int h = 0; h < num_heads; h++) {
            const int head_offset = h * head_size;

            for (int i = 0; i < seq_len; i++) {
                const float* q_row = Q_data + i * d_model + head_offset;
                const float* k_row = K_data + i * d_model + head_offset;
                const float* v_row = V_data + i * d_model + head_offset;

                float* q_out = Q_head_data + i * head_size;
                float* k_out = K_head_data + i * head_size;
                float* v_out = V_head_data + i * head_size;

                std::memcpy(q_out, q_row, head_size * sizeof(float));
                std::memcpy(k_out, k_row, head_size * sizeof(float));
                std::memcpy(v_out, v_row, head_size * sizeof(float));
            }

            blas_sgemm(Q_head_data, K_head_data, scores_data,
                      seq_len, seq_len, head_size, false, true);

            for (int i = 0; i < seq_len * seq_len; i++) {
                scores_data[i] = scores_data[i] * scale_factor + mask_data[i];
            }

            Tensor attention_weights = scores.softmax();

            if (training && dropout_rate > 0.0f) {
                attention_weights = dropout(attention_weights, dropout_rate, training);
            }

            Tensor attended(seq_len, head_size);
            blas_sgemm(attention_weights.raw(), V_head_data, attended.raw(),
                      seq_len, head_size, seq_len, false, false);

            const float* attended_data = attended.raw();
            for (int i = 0; i < seq_len; i++) {
                float* out_row = result_data + i * d_model + head_offset;
                const float* in_row = attended_data + i * head_size;
                std::memcpy(out_row, in_row, head_size * sizeof(float));
            }
        }

        auto concat_var = Variable::create(result, input->requiresGrad());
        auto self_concat = concat_var;
        auto output = concat_var->matmul(W_o)->add(b_o);

        if (training && dropout_rate > 0.0f) {
            output = output->dropout(dropout_rate, training);
        }

        if (input->requiresGrad()) {
            output->addChild(input);
            output->addChild(W_q);
            output->addChild(W_k);
            output->addChild(W_v);
            output->addChild(W_o);

            auto self_bq = b_q;
            auto self_bk = b_k;
            auto self_bv = b_v;
            auto self_bo = b_o;

            output->setBackwardFn([self_input, self_Q, self_K, self_V, self_Wq, self_Wk, self_Wv, self_Wo,
                                   self_bq, self_bk, self_bv, self_bo,
                                   self_concat, output, self_num_heads,
                                   self_d_model, seq_len, head_size, causal_mask, scale_factor]() {
                if (!output->hasGrad()) return;
                self_Wq->ensureGrad(); self_Wk->ensureGrad();
                self_Wv->ensureGrad(); self_Wo->ensureGrad();
                self_bq->ensureGrad(); self_bk->ensureGrad();
                self_bv->ensureGrad(); self_bo->ensureGrad();
                self_input->ensureGrad();

                self_Wo->getGrad().add_inplace(self_concat->getData().transpose().matmul(output->getGrad()));

                Tensor db_o(1, self_d_model);
                db_o.fill(0.0f);
                float* db_o_data = db_o.raw();
                const float* output_grad_data = output->getGrad().raw();

                for (int i = 0; i < seq_len; i++) {
                    for (int j = 0; j < self_d_model; j++) {
                        db_o_data[j] += output_grad_data[i * self_d_model + j];
                    }
                }
                self_bo->getGrad().add_inplace(db_o);

                Tensor dConcat = output->getGrad().matmul(self_Wo->getData().transpose());

                Tensor dQ(seq_len, self_d_model);
                Tensor dK(seq_len, self_d_model);
                Tensor dV(seq_len, self_d_model);
                dQ.fill(0.0f);
                dK.fill(0.0f);
                dV.fill(0.0f);

                const float* Q_data = self_Q->getData().raw();
                const float* K_data = self_K->getData().raw();
                const float* V_data = self_V->getData().raw();
                const float* dConcat_data = dConcat.raw();
                const float* mask_data = causal_mask.raw();
                float* dQ_data = dQ.raw();
                float* dK_data = dK.raw();
                float* dV_data = dV.raw();

                Tensor Q_head(seq_len, head_size);
                Tensor K_head(seq_len, head_size);
                Tensor V_head(seq_len, head_size);
                Tensor dAttended(seq_len, head_size);
                Tensor scores(seq_len, seq_len);
                Tensor dScores(seq_len, seq_len);

                float* Q_head_data = Q_head.raw();
                float* K_head_data = K_head.raw();
                float* V_head_data = V_head.raw();
                float* dAttended_data = dAttended.raw();
                float* scores_data = scores.raw();
                float* dScores_data = dScores.raw();

                for (int h = 0; h < self_num_heads; h++) {
                    const int start_col = h * head_size;

                    for (int i = 0; i < seq_len; i++) {
                        std::memcpy(dAttended_data + i * head_size,
                                   dConcat_data + i * self_d_model + start_col,
                                   head_size * sizeof(float));
                        std::memcpy(Q_head_data + i * head_size,
                                   Q_data + i * self_d_model + start_col,
                                   head_size * sizeof(float));
                        std::memcpy(K_head_data + i * head_size,
                                   K_data + i * self_d_model + start_col,
                                   head_size * sizeof(float));
                        std::memcpy(V_head_data + i * head_size,
                                   V_data + i * self_d_model + start_col,
                                   head_size * sizeof(float));
                    }

                    blas_sgemm(Q_head_data, K_head_data, scores_data,
                              seq_len, seq_len, head_size, false, true);

                    for (int i = 0; i < seq_len * seq_len; i++) {
                        scores_data[i] = scores_data[i] * scale_factor + mask_data[i];
                    }

                    Tensor attn_weights = scores.softmax();
                    const float* attn_data = attn_weights.raw();

                    Tensor dAttnWeights = dAttended.matmul(V_head.transpose());
                    Tensor dV_head = attn_weights.transpose().matmul(dAttended);
                    const float* dAttnWeights_data = dAttnWeights.raw();

                    for (int i = 0; i < seq_len; i++) {
                        float sum = 0.0f;
                        for (int j = 0; j < seq_len; j++) {
                            sum += dAttnWeights_data[i * seq_len + j] * attn_data[i * seq_len + j];
                        }
                        for (int j = 0; j < seq_len; j++) {
                            dScores_data[i * seq_len + j] = attn_data[i * seq_len + j] *
                                (dAttnWeights_data[i * seq_len + j] - sum) * scale_factor;
                        }
                    }

                    Tensor dQ_head = dScores.matmul(K_head);
                    Tensor dK_head = dScores.transpose().matmul(Q_head);

                    const float* dQ_head_data = dQ_head.raw();
                    const float* dK_head_data = dK_head.raw();
                    const float* dV_head_data = dV_head.raw();

                    for (int i = 0; i < seq_len; i++) {
                        for (int j = 0; j < head_size; j++) {
                            dQ_data[i * self_d_model + start_col + j] += dQ_head_data[i * head_size + j];
                            dK_data[i * self_d_model + start_col + j] += dK_head_data[i * head_size + j];
                            dV_data[i * self_d_model + start_col + j] += dV_head_data[i * head_size + j];
                        }
                    }
                }

                self_Wq->getGrad().add_inplace(self_input->getData().transpose().matmul(dQ));
                self_Wk->getGrad().add_inplace(self_input->getData().transpose().matmul(dK));
                self_Wv->getGrad().add_inplace(self_input->getData().transpose().matmul(dV));

                Tensor db_q(1, self_d_model);
                Tensor db_k(1, self_d_model);
                Tensor db_v(1, self_d_model);
                db_q.fill(0.0f);
                db_k.fill(0.0f);
                db_v.fill(0.0f);

                float* db_q_data = db_q.raw();
                float* db_k_data = db_k.raw();
                float* db_v_data = db_v.raw();

                for (int i = 0; i < seq_len; i++) {
                    for (int j = 0; j < self_d_model; j++) {
                        db_q_data[j] += dQ_data[i * self_d_model + j];
                        db_k_data[j] += dK_data[i * self_d_model + j];
                        db_v_data[j] += dV_data[i * self_d_model + j];
                    }
                }

                self_bq->getGrad().add_inplace(db_q);
                self_bk->getGrad().add_inplace(db_k);
                self_bv->getGrad().add_inplace(db_v);

                Tensor dInput = dQ.matmul(self_Wq->getData().transpose())
                               .add(dK.matmul(self_Wk->getData().transpose()))
                               .add(dV.matmul(self_Wv->getData().transpose()));
                self_input->getGrad().add_inplace(dInput);
            });
        }

        return output;

    } else {
        // 3D case: (batch, seq, d_model). This is the training path.
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
        auto Q = std::make_shared<Tensor>(flat, d);
        auto K = std::make_shared<Tensor>(flat, d);
        auto V = std::make_shared<Tensor>(flat, d);
        auto concat = std::make_shared<Tensor>(flat, d);

        const float* in_data = input_tensor.raw();
        auto project = [&](const Tensor& W, const Tensor& b, Tensor& out) {
            fill_bias_rows(out.raw(), b.raw(), flat, d);
            blas_sgemm_ex(in_data, W.raw(), out.raw(), flat, d, d,
                          false, false, 1.0f, 1.0f);
        };
        project(W_q->getData(), b_q->getData(), *Q);
        project(W_k->getData(), b_k->getData(), *K);
        project(W_v->getData(), b_v->getData(), *V);

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

        parallel_for(static_cast<size_t>(batch_size) * H, 1,
                     [&](size_t begin, size_t end) {
            Tensor Q_head(S, head_size);
            Tensor K_head(S, head_size);
            Tensor V_head(S, head_size);
            Tensor scores(S, S);

            for (size_t unit = begin; unit < end; unit++) {
                const int b = static_cast<int>(unit) / H;
                const int h = static_cast<int>(unit) % H;
                const size_t batch_offset = static_cast<size_t>(b) * S * d;
                const int col = h * head_size;

                gather_head(Q_data + batch_offset + col, Q_head.raw(), S, d, head_size);
                gather_head(K_data + batch_offset + col, K_head.raw(), S, d, head_size);
                gather_head(V_data + batch_offset + col, V_head.raw(), S, d, head_size);

                float* scores_data = scores.raw();
                blas_sgemm(Q_head.raw(), K_head.raw(), scores_data,
                           S, S, head_size, false, true);
                for (int i = 0; i < S * S; i++) {
                    scores_data[i] = scores_data[i] * scale_factor + mask_data[i];
                }

                // Nested parallel_for inside softmax runs inline here.
                Tensor weights = scores.softmax();
                float* attn_unit = attn_data_all + unit * S * S;
                std::memcpy(attn_unit, weights.raw(), sizeof(float) * S * S);

                const float* effective = attn_unit;
                Tensor dropped(1, 1);
                if (use_attn_dropout) {
                    float* m = mask_all + unit * S * S;
                    fill_dropout_mask(m, static_cast<size_t>(S) * S,
                                      dropout_rate, keep_scale);
                    dropped = Tensor(S, S);
                    float* w = dropped.raw();
                    for (int i = 0; i < S * S; i++) {
                        w[i] = attn_unit[i] * m[i];
                    }
                    effective = w;
                }

                Tensor attended(S, head_size);
                blas_sgemm(effective, V_head.raw(), attended.raw(),
                           S, head_size, S, false, false);

                scatter_head(attended.raw(), concat_data + batch_offset + col,
                             S, d, head_size);
            }
        });

        // Output projection, also flat.
        Tensor out_tensor(batch_size, S, d);
        fill_bias_rows(out_tensor.raw(), b_o->getData().raw(), flat, d);
        blas_sgemm_ex(concat_data, W_o->getData().raw(), out_tensor.raw(),
                      flat, d, d, false, false, 1.0f, 1.0f);

        bool needs_grad = input->requiresGrad();
        auto output = Variable::create(out_tensor, needs_grad);

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
            output->setBackwardFn([self_input, self_Wq, self_Wk, self_Wv, self_Wo,
                                   self_bq, self_bk, self_bv, self_bo,
                                   Q, K, V, concat, attn_cache, drop_mask,
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
                Tensor dConcat(flat, d);
                blas_sgemm_ex(dOut, self_Wo->getData().raw(), dConcat.raw(),
                              flat, d, d, false, true, 1.0f, 0.0f);

                Tensor dQ(flat, d);
                Tensor dK(flat, d);
                Tensor dV(flat, d);
                float* dQ_data = dQ.raw();
                float* dK_data = dK.raw();
                float* dV_data = dV.raw();
                const float* dConcat_data = dConcat.raw();
                const float* Q_data = Q->raw();
                const float* K_data = K->raw();
                const float* V_data = V->raw();
                const float* attn_all = attn_cache->data();
                const float* mask_all = drop_mask ? drop_mask->data() : nullptr;

                parallel_for(static_cast<size_t>(batch_size) * H, 1,
                             [&](size_t begin, size_t end) {
                    Tensor Q_head(S, head_size);
                    Tensor K_head(S, head_size);
                    Tensor V_head(S, head_size);
                    Tensor dAttended(S, head_size);
                    Tensor dWeights(S, S);
                    Tensor dScores(S, S);
                    Tensor dQ_head(S, head_size);
                    Tensor dK_head(S, head_size);
                    Tensor dV_head(S, head_size);
                    Tensor effective(S, S);

                    for (size_t unit = begin; unit < end; unit++) {
                        const int b = static_cast<int>(unit) / H;
                        const int h = static_cast<int>(unit) % H;
                        const size_t batch_offset = static_cast<size_t>(b) * S * d;
                        const int col = h * head_size;

                        gather_head(Q_data + batch_offset + col, Q_head.raw(), S, d, head_size);
                        gather_head(K_data + batch_offset + col, K_head.raw(), S, d, head_size);
                        gather_head(V_data + batch_offset + col, V_head.raw(), S, d, head_size);
                        gather_head(dConcat_data + batch_offset + col, dAttended.raw(), S, d, head_size);

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
                        blas_sgemm(dAttended.raw(), V_head.raw(), dWeights.raw(),
                                   S, S, head_size, false, true);
                        blas_sgemm(W_eff, dAttended.raw(), dV_head.raw(),
                                   S, head_size, S, true, false);

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
                        blas_sgemm(dS, K_head.raw(), dQ_head.raw(),
                                   S, head_size, S, false, false);
                        blas_sgemm(dS, Q_head.raw(), dK_head.raw(),
                                   S, head_size, S, true, false);

                        scatter_head(dQ_head.raw(), dQ_data + batch_offset + col, S, d, head_size);
                        scatter_head(dK_head.raw(), dK_data + batch_offset + col, S, d, head_size);
                        scatter_head(dV_head.raw(), dV_data + batch_offset + col, S, d, head_size);
                    }
                });

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
}
