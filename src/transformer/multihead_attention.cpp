#include "transformer/tensor.h"
#include "transformer/activations.h"
#include "transformer/multihead_attention.h"
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cstring>

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

        std::vector<Tensor> attention_weights_all;
        std::vector<Tensor> V_heads_all;

        const float* Q_data = Q->getData().raw();
        const float* K_data = K->getData().raw();
        const float* V_data = V->getData().raw();
        float* result_data = result.raw();

        Tensor causal_mask = Tensor::create_causal_mask(seq_len);
        const float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_size));

        for (int h = 0; h < num_heads; h++) {
            int head_offset = h * head_size;

            Tensor scores(seq_len, seq_len);
            scores.fill(0.0f);
            float* scores_data = scores.raw();

            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    float sum = 0.0f;
                    
                    for (int k = 0; k < head_size; k++) {
                        sum += Q_data[i * d_model + head_offset + k] * 
                               K_data[j * d_model + head_offset + k];
                    }
                    
                    scores_data[i * seq_len + j] = sum * scale_factor + causal_mask.getValue(i, j);
                }
            }

            Tensor attention_weights = scores.softmax();
            
            if (training && dropout_rate > 0.0f) {
                attention_weights = dropout(attention_weights, dropout_rate, training);
            }

            Tensor V_head(seq_len, head_size);
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < head_size; j++) {
                    V_head.setValue(i, j, V_data[i * d_model + head_offset + j]);
                }
            }

            attention_weights_all.push_back(attention_weights);
            V_heads_all.push_back(V_head);

            const float* attn_data = attention_weights.raw();

            for (int i = 0; i < seq_len; i++) {
                for (int k = 0; k < head_size; k++) {
                    float sum = 0.0f;
                    
                    for (int j = 0; j < seq_len; j++) {
                        sum += attn_data[i * seq_len + j] * V_head.getValue(j, k);
                    }
                    
                    result_data[i * d_model + head_offset + k] = sum;
                }
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
                                   attention_weights_all, V_heads_all, self_concat, output, self_num_heads,
                                   self_d_model, seq_len, head_size]() {

                self_Wo->getGrad().add_inplace(self_concat->getData().transpose().matmul(output->getGrad()));

                Tensor db_o(1, self_d_model);
                db_o.fill(0.0f);
                for (int i = 0; i < seq_len; i++) {
                    for (int j = 0; j < self_d_model; j++) {
                        db_o.setValue(0, j, db_o.getValue(0, j) + output->getGrad().getValue(i, j));
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

                for (int h = 0; h < self_num_heads; h++) {
                    int start_col = h * head_size;
                    
                    Tensor dAttended(seq_len, head_size);
                    for (int i = 0; i < seq_len; i++) {
                        for (int j = 0; j < head_size; j++) {
                            dAttended.setValue(i, j, dConcat.getValue(i, start_col + j));
                        }
                    }

                    const Tensor& attn_weights = attention_weights_all[h];
                    const Tensor& V_head = V_heads_all[h];

                    Tensor dAttnWeights = dAttended.matmul(V_head.transpose());
                    Tensor dV_head = attn_weights.transpose().matmul(dAttended);

                    Tensor dScores(seq_len, seq_len);
                    for (int i = 0; i < seq_len; i++) {
                        float sum = 0.0f;
                        for (int j = 0; j < seq_len; j++) {
                            sum += dAttnWeights.getValue(i, j) * attn_weights.getValue(i, j);
                        }
                        for (int j = 0; j < seq_len; j++) {
                            float grad = attn_weights.getValue(i, j) * (dAttnWeights.getValue(i, j) - sum);
                            dScores.setValue(i, j, grad);
                        }
                    }

                    dScores.scale_inplace(1.0f / std::sqrt(head_size));

                    Tensor Q_head = self_Q->getData().slice(0, seq_len, start_col, head_size);
                    Tensor K_head = self_K->getData().slice(0, seq_len, start_col, head_size);

                    Tensor dQ_head = dScores.matmul(K_head);
                    Tensor dK_head = dScores.transpose().matmul(Q_head);

                    for (int i = 0; i < seq_len; i++) {
                        for (int j = 0; j < head_size; j++) {
                            dQ.setValue(i, start_col + j, dQ.getValue(i, start_col + j) + dQ_head.getValue(i, j));
                            dK.setValue(i, start_col + j, dK.getValue(i, start_col + j) + dK_head.getValue(i, j));
                            dV.setValue(i, start_col + j, dV.getValue(i, start_col + j) + dV_head.getValue(i, j));
                        }
                    }
                }

                Tensor input_T = self_input->getData().transpose();
                self_Wq->getGrad().add_inplace(input_T.matmul(dQ));
                self_Wk->getGrad().add_inplace(input_T.matmul(dK));
                self_Wv->getGrad().add_inplace(input_T.matmul(dV));

                Tensor db_q(1, self_d_model);
                Tensor db_k(1, self_d_model);
                Tensor db_v(1, self_d_model);
                db_q.fill(0.0f);
                db_k.fill(0.0f);
                db_v.fill(0.0f);

                for (int i = 0; i < seq_len; i++) {
                    for (int j = 0; j < self_d_model; j++) {
                        db_q.setValue(0, j, db_q.getValue(0, j) + dQ.getValue(i, j));
                        db_k.setValue(0, j, db_k.getValue(0, j) + dK.getValue(i, j));
                        db_v.setValue(0, j, db_v.getValue(0, j) + dV.getValue(i, j));
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
        int batch_size = input_tensor.getBatchSize();
        int seq_len = input_tensor.getRows();

        // Compute Q, K, V as raw Tensors to avoid automatic backward conflicting with custom backward
        Tensor Q_data = input_tensor.matmul(W_q->getData()).add(b_q->getData());
        Tensor K_data = input_tensor.matmul(W_k->getData()).add(b_k->getData());
        Tensor V_data = input_tensor.matmul(W_v->getData()).add(b_v->getData());

        // Wrap in Variables for the backward pass to access
        auto Q = Variable::create(Q_data, false);  // Don't track gradients
        auto K = Variable::create(K_data, false);
        auto V = Variable::create(V_data, false);

        int head_size = d_model / num_heads;
        Tensor result(batch_size, seq_len, d_model);
        Tensor causal_mask = Tensor::create_causal_mask_batch(batch_size, seq_len);

        std::vector<Tensor> attention_weights_all;
        std::vector<Tensor> V_heads_all;

        for (int i = 0; i < num_heads; i++) {
            int start_col = i * head_size;

            Tensor Q_head(batch_size, seq_len, head_size);
            Tensor K_head(batch_size, seq_len, head_size);
            Tensor V_head(batch_size, seq_len, head_size);

            for (int b = 0; b < batch_size; b++) {
                for (int s = 0; s < seq_len; s++) {
                    for (int h = 0; h < head_size; h++) {
                        Q_head.setValue(b, s, h, Q->getData().getValue(b, s, start_col + h));
                        K_head.setValue(b, s, h, K->getData().getValue(b, s, start_col + h));
                        V_head.setValue(b, s, h, V->getData().getValue(b, s, start_col + h));
                    }
                }
            }

            Tensor K_transposed = K_head.transpose();
            Tensor scores = Q_head.matmul(K_transposed);
            scores.scale_inplace(1.0f / std::sqrt(head_size));
            scores.add_inplace(causal_mask);

            Tensor attention_weights = scores.softmax();

            attention_weights_all.push_back(attention_weights);
            V_heads_all.push_back(V_head);

            attention_weights = dropout(attention_weights, dropout_rate, training);
            Tensor attended_values = attention_weights.matmul(V_head);

            for (int b = 0; b < batch_size; b++) {
                for (int s = 0; s < seq_len; s++) {
                    for (int h = 0; h < head_size; h++) {
                        result.setValue(b, s, start_col + h, attended_values.getValue(b, s, h));
                    }
                }
            }
        }

        // Compute output as raw Tensor to avoid automatic backward for b_o
        Tensor concat_result = result;
        Tensor output_data = concat_result.matmul(W_o->getData()).add(b_o->getData());

        auto concat_var = Variable::create(concat_result, false);  // Don't track gradients
        auto output = Variable::create(output_data, input->requiresGrad());

        if (training && dropout_rate > 0.0f) {
            output = output->dropout(dropout_rate, training);
        }

        if (input->requiresGrad()) {
            output->addChild(input);
            output->addChild(W_q);
            output->addChild(W_k);
            output->addChild(W_v);
            output->addChild(W_o);

            auto self_input = input;
            auto self_Q = Q;
            auto self_K = K;
            auto self_V = V;
            auto self_Wq = W_q;
            auto self_Wk = W_k;
            auto self_Wv = W_v;
            auto self_Wo = W_o;
            auto self_bq = b_q;
            auto self_bk = b_k;
            auto self_bv = b_v;
            auto self_bo = b_o;
            auto self_concat = concat_var;
            int self_num_heads = num_heads;
            int self_d_model = d_model;
            size_t self_batch_size = batch_size;

            output->setBackwardFn([self_input, self_Q, self_K, self_V, self_Wq, self_Wk, self_Wv, self_Wo,
                                   self_bq, self_bk, self_bv, self_bo,
                                   attention_weights_all, V_heads_all, self_concat, output, self_num_heads,
                                   self_d_model, self_batch_size, seq_len, head_size]() {

                for (size_t b = 0; b < self_batch_size; b++) {
                    Tensor concat_slice(seq_len, self_d_model);
                    Tensor output_grad_slice(seq_len, self_d_model);
                    for (int i = 0; i < seq_len; i++) {
                        for (int j = 0; j < self_d_model; j++) {
                            concat_slice.setValue(i, j, self_concat->getData().getValue(b, i, j));
                            output_grad_slice.setValue(i, j, output->getGrad().getValue(b, i, j));
                        }
                    }
                    self_Wo->getGrad().add_inplace(concat_slice.transpose().matmul(output_grad_slice));
                }

                Tensor db_o(1, self_d_model);
                db_o.fill(0.0f);
                for (size_t b = 0; b < self_batch_size; b++) {
                    for (int i = 0; i < seq_len; i++) {
                        for (int j = 0; j < self_d_model; j++) {
                            db_o.setValue(0, j, db_o.getValue(0, j) + output->getGrad().getValue(b, i, j));
                        }
                    }
                }
                self_bo->getGrad().add_inplace(db_o);

                Tensor dConcat = output->getGrad().matmul(self_Wo->getData().transpose());

                Tensor dQ(self_batch_size, seq_len, self_d_model);
                Tensor dK(self_batch_size, seq_len, self_d_model);
                Tensor dV(self_batch_size, seq_len, self_d_model);
                dQ.fill(0.0f);
                dK.fill(0.0f);
                dV.fill(0.0f);

                for (size_t b = 0; b < self_batch_size; b++) {
                    for (int h = 0; h < self_num_heads; h++) {
                        int start_col = h * head_size;

                        Tensor dAttended(seq_len, head_size);
                        for (int i = 0; i < seq_len; i++) {
                            for (int j = 0; j < head_size; j++) {
                                dAttended.setValue(i, j, dConcat.getValue(b, i, start_col + j));
                            }
                        }
                        // Extract 2D slices from 3D tensors for this batch element
                        // attention_weights_all[h] is (batch_size, seq_len, seq_len)
                        // V_heads_all[h] is (batch_size, seq_len, head_size)
                        const Tensor& attn_weights_3d = attention_weights_all[h];
                        const Tensor& V_head_3d = V_heads_all[h];

                        Tensor attn_weights(seq_len, seq_len);
                        Tensor V_head(seq_len, head_size);

                        for (int i = 0; i < seq_len; i++) {
                            for (int j = 0; j < seq_len; j++) {
                                attn_weights.setValue(i, j, attn_weights_3d.getValue(b, i, j));
                            }
                            for (int j = 0; j < head_size; j++) {
                                V_head.setValue(i, j, V_head_3d.getValue(b, i, j));
                            }
                        }

                        Tensor dAttnWeights = dAttended.matmul(V_head.transpose());
                        Tensor dV_head = attn_weights.transpose().matmul(dAttended);

                        Tensor dScores(seq_len, seq_len);
                        for (int i = 0; i < seq_len; i++) {
                            float sum = 0.0f;
                            for (int j = 0; j < seq_len; j++) {
                                sum += dAttnWeights.getValue(i, j) * attn_weights.getValue(i, j);
                            }
                            for (int j = 0; j < seq_len; j++) {
                                float grad = attn_weights.getValue(i, j) * (dAttnWeights.getValue(i, j) - sum);
                                dScores.setValue(i, j, grad);
                            }
                        }

                        dScores.scale_inplace(1.0f / std::sqrt(static_cast<float>(head_size)));

                        Tensor Q_head_b(seq_len, head_size);
                        Tensor K_head_b(seq_len, head_size);
                        for (int i = 0; i < seq_len; i++) {
                            for (int j = 0; j < head_size; j++) {
                                Q_head_b.setValue(i, j, self_Q->getData().getValue(b, i, start_col + j));
                                K_head_b.setValue(i, j, self_K->getData().getValue(b, i, start_col + j));
                            }
                        }

                        Tensor dQ_head = dScores.matmul(K_head_b);
                        Tensor dK_head = dScores.transpose().matmul(Q_head_b);

                        for (int i = 0; i < seq_len; i++) {
                            for (int j = 0; j < head_size; j++) {
                                dQ.setValue(b, i, start_col + j,
                                           dQ.getValue(b, i, start_col + j) + dQ_head.getValue(i, j));
                                dK.setValue(b, i, start_col + j,
                                           dK.getValue(b, i, start_col + j) + dK_head.getValue(i, j));
                                dV.setValue(b, i, start_col + j,
                                           dV.getValue(b, i, start_col + j) + dV_head.getValue(i, j));
                            }
                        }
                    }
                }

                for (size_t b = 0; b < self_batch_size; b++) {
                    Tensor input_slice(seq_len, self_d_model);
                    for (int i = 0; i < seq_len; i++) {
                        for (int j = 0; j < self_d_model; j++) {
                            input_slice.setValue(i, j, self_input->getData().getValue(b, i, j));
                        }
                    }
                    Tensor input_T = input_slice.transpose();

                    Tensor dQ_slice(seq_len, self_d_model);
                    Tensor dK_slice(seq_len, self_d_model);
                    Tensor dV_slice(seq_len, self_d_model);
                    for (int i = 0; i < seq_len; i++) {
                        for (int j = 0; j < self_d_model; j++) {
                            dQ_slice.setValue(i, j, dQ.getValue(b, i, j));
                            dK_slice.setValue(i, j, dK.getValue(b, i, j));
                            dV_slice.setValue(i, j, dV.getValue(b, i, j));
                        }
                    }

                    self_Wq->getGrad().add_inplace(input_T.matmul(dQ_slice));
                    self_Wk->getGrad().add_inplace(input_T.matmul(dK_slice));
                    self_Wv->getGrad().add_inplace(input_T.matmul(dV_slice));
                }

                Tensor db_q(1, self_d_model);
                Tensor db_k(1, self_d_model);
                Tensor db_v(1, self_d_model);
                db_q.fill(0.0f);
                db_k.fill(0.0f);
                db_v.fill(0.0f);

                for (size_t b = 0; b < self_batch_size; b++) {
                    for (int i = 0; i < seq_len; i++) {
                        for (int j = 0; j < self_d_model; j++) {
                            db_q.setValue(0, j, db_q.getValue(0, j) + dQ.getValue(b, i, j));
                            db_k.setValue(0, j, db_k.getValue(0, j) + dK.getValue(b, i, j));
                            db_v.setValue(0, j, db_v.getValue(0, j) + dV.getValue(b, i, j));
                        }
                    }
                }

                self_bq->getGrad().add_inplace(db_q);
                self_bk->getGrad().add_inplace(db_k);
                self_bv->getGrad().add_inplace(db_v);

                Tensor dInput(self_batch_size, seq_len, self_d_model);
                for (size_t b = 0; b < self_batch_size; b++) {
                    Tensor dQ_slice(seq_len, self_d_model);
                    Tensor dK_slice(seq_len, self_d_model);
                    Tensor dV_slice(seq_len, self_d_model);
                    for (int i = 0; i < seq_len; i++) {
                        for (int j = 0; j < self_d_model; j++) {
                            dQ_slice.setValue(i, j, dQ.getValue(b, i, j));
                            dK_slice.setValue(i, j, dK.getValue(b, i, j));
                            dV_slice.setValue(i, j, dV.getValue(b, i, j));
                        }
                    }

                    Tensor dInput_slice = dQ_slice.matmul(self_Wq->getData().transpose())
                                         .add(dK_slice.matmul(self_Wk->getData().transpose()))
                                         .add(dV_slice.matmul(self_Wv->getData().transpose()));

                    for (int i = 0; i < seq_len; i++) {
                        for (int j = 0; j < self_d_model; j++) {
                            dInput.setValue(b, i, j, dInput_slice.getValue(i, j));
                        }
                    }
                }
                self_input->getGrad().add_inplace(dInput);
            });
        }

        return output;
    }
}