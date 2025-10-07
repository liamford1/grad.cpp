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
}

MultiHeadAttention::~MultiHeadAttention() {}

std::shared_ptr<Variable> MultiHeadAttention::forward(std::shared_ptr<Variable> input, bool training) const {
    const Tensor& input_tensor = input->getData();

    if (!input_tensor.getIs3D()) {
        int seq_len = input_tensor.getRows();
        int head_size = d_model / num_heads;

        auto Q = input->matmul(W_q);
        auto K = input->matmul(W_k);
        auto V = input->matmul(W_v);

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
        auto output = concat_var->matmul(W_o);

        if (training && dropout_rate > 0.0f) {
            Tensor output_dropped = dropout(output->getData(), dropout_rate, training);
            output = Variable::create(output_dropped, input->requiresGrad());
        }

        if (input->requiresGrad()) {
            output->addChild(input);
            output->addChild(W_q);
            output->addChild(W_k);
            output->addChild(W_v);
            output->addChild(W_o);

            output->setBackwardFn([self_input, self_Q, self_K, self_V, self_Wq, self_Wk, self_Wv, self_Wo, 
                                   attention_weights_all, V_heads_all, output, self_num_heads, 
                                   self_d_model, seq_len, head_size]() {
                
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

                    dScores = dScores.scale(1.0f / std::sqrt(head_size));

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
                self_Wq->getGrad() = self_Wq->getGrad().add(input_T.matmul(dQ));
                self_Wk->getGrad() = self_Wk->getGrad().add(input_T.matmul(dK));
                self_Wv->getGrad() = self_Wv->getGrad().add(input_T.matmul(dV));

                Tensor dInput = dQ.matmul(self_Wq->getData().transpose())
                               .add(dK.matmul(self_Wk->getData().transpose()))
                               .add(dV.matmul(self_Wv->getData().transpose()));
                self_input->getGrad() = self_input->getGrad().add(dInput);

                Tensor concat_result(seq_len, self_d_model);
                for (int h = 0; h < self_num_heads; h++) {
                    int start_col = h * head_size;
                    const Tensor& attn_weights = attention_weights_all[h];
                    const Tensor& V_head = V_heads_all[h];
                    Tensor attended = attn_weights.matmul(V_head);
                    for (int i = 0; i < seq_len; i++) {
                        for (int j = 0; j < head_size; j++) {
                            concat_result.setValue(i, start_col + j, attended.getValue(i, j));
                        }
                    }
                }

                self_Wo->getGrad() = self_Wo->getGrad().add(concat_result.transpose().matmul(output->getGrad()));
            });
        }

        return output;
    } else {
        int batch_size = input_tensor.getBatchSize();
        int seq_len = input_tensor.getRows();

        auto Q = input->matmul(W_q);
        auto K = input->matmul(W_k);
        auto V = input->matmul(W_v);

        int head_size = d_model / num_heads;
        Tensor result(batch_size, seq_len, d_model);
        Tensor causal_mask = Tensor::create_causal_mask_batch(batch_size, seq_len);

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
            scores = scores.scale(1.0f / std::sqrt(head_size));
            scores = scores.add(causal_mask);

            Tensor attention_weights = scores.softmax();
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

        auto concat_var = Variable::create(result, input->requiresGrad());
        auto output = concat_var->matmul(W_o);
        
        if (training && dropout_rate > 0.0f) {
            Tensor output_dropped = dropout(output->getData(), dropout_rate, training);
            output = Variable::create(output_dropped, input->requiresGrad());
        }

        return output;
    }
}