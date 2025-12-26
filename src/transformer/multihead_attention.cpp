#include "transformer/tensor.h"
#include "transformer/activations.h"
#include "transformer/multihead_attention.h"
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cstring>
#include "transformer/blas_wrapper.h"

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

    std::cerr << "[DEBUG] MHA: Input is3D=" << input_tensor.getIs3D()
              << ", rows=" << input_tensor.getRows()
              << ", cols=" << input_tensor.getCols()
              << ", device=" << (input_tensor.getDevice() == Device::CUDA ? "CUDA" : "CPU") << std::endl;

    if (!input_tensor.getIs3D()) {
        std::cerr << "[DEBUG] MHA: Entering 2D case" << std::endl;
        int seq_len = input_tensor.getRows();
        int head_size = d_model / num_heads;

        std::cerr << "[DEBUG] MHA: Transferring input to CPU before Q, K, V projections" << std::endl;
        Tensor input_cpu = (input_tensor.getDevice() == Device::CUDA) ? input_tensor.to(Device::CPU) : input_tensor;
        auto input_cpu_var = Variable::create(input_cpu, input->requiresGrad());

        std::cerr << "[DEBUG] MHA: Computing Q, K, V projections" << std::endl;
        auto Q = input_cpu_var->matmul(W_q)->add(b_q);
        auto K = input_cpu_var->matmul(W_k)->add(b_k);
        auto V = input_cpu_var->matmul(W_v)->add(b_v);

        std::cerr << "[DEBUG] MHA: Q device=" << (Q->getData().getDevice() == Device::CUDA ? "CUDA" : "CPU") << std::endl;
        std::cerr << "[DEBUG] MHA: Transferring Q, K, V to CPU" << std::endl;
        Tensor Q_cpu = (Q->getData().getDevice() == Device::CUDA) ? Q->getData().to(Device::CPU) : Q->getData();
        Tensor K_cpu = (K->getData().getDevice() == Device::CUDA) ? K->getData().to(Device::CPU) : K->getData();
        Tensor V_cpu = (V->getData().getDevice() == Device::CUDA) ? V->getData().to(Device::CPU) : V->getData();
        std::cerr << "[DEBUG] MHA: Transfer complete, Q_cpu device=" << (Q_cpu.getDevice() == Device::CUDA ? "CUDA" : "CPU") << std::endl;

        Tensor result(seq_len, d_model, Device::CPU);
        result.fill(0.0f);

        auto self_input = input;
        auto self_input_cpu = input_cpu_var;
        auto self_Q = Q;
        auto self_K = K;
        auto self_V = V;
        auto self_Wq = W_q;
        auto self_Wk = W_k;
        auto self_Wv = W_v;
        auto self_Wo = W_o;
        int self_num_heads = num_heads;
        int self_d_model = d_model;

        const float* Q_data = Q_cpu.raw();
        const float* K_data = K_cpu.raw();
        const float* V_data = V_cpu.raw();
        float* result_data = result.raw();

        Tensor causal_mask = Tensor::create_causal_mask(seq_len);
        const float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_size));
        const float* mask_data = causal_mask.raw();

        Tensor Q_head(seq_len, head_size, Device::CPU);
        Tensor K_head(seq_len, head_size, Device::CPU);
        Tensor V_head(seq_len, head_size, Device::CPU);
        Tensor scores(seq_len, seq_len, Device::CPU);

        std::cerr << "[DEBUG] MHA: Getting raw pointers (2D)" << std::endl;
        float* Q_head_data = Q_head.raw();
        float* K_head_data = K_head.raw();
        float* V_head_data = V_head.raw();
        float* scores_data = scores.raw();

        std::cerr << "[DEBUG] MHA: Starting attention computation for " << num_heads << " heads (2D)" << std::endl;
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

            Tensor attended(seq_len, head_size, Device::CPU);
            blas_sgemm(attention_weights.raw(), V_head_data, attended.raw(),
                      seq_len, head_size, seq_len, false, false);

            const float* attended_data = attended.raw();
            for (int i = 0; i < seq_len; i++) {
                float* out_row = result_data + i * d_model + head_offset;
                const float* in_row = attended_data + i * head_size;
                std::memcpy(out_row, in_row, head_size * sizeof(float));
            }
        }

        std::cerr << "[DEBUG] MHA: Attention computation complete" << std::endl;
        std::cerr << "[DEBUG] MHA: Transferring result back to " << (input_tensor.getDevice() == Device::CUDA ? "CUDA" : "CPU") << std::endl;
        Tensor result_device = (input_tensor.getDevice() == Device::CUDA) ? result.to(Device::CUDA) : result;
        std::cerr << "[DEBUG] MHA: Creating output variable" << std::endl;
        auto concat_var = Variable::create(result_device, input->requiresGrad());
        auto self_concat = concat_var;
        std::cerr << "[DEBUG] MHA: Computing final linear projection" << std::endl;
        auto output = concat_var->matmul(W_o)->add(b_o);

        if (training && dropout_rate > 0.0f) {
            std::cerr << "[DEBUG] MHA: Applying dropout" << std::endl;
            output = output->dropout(dropout_rate, training);
        }
        std::cerr << "[DEBUG] MHA: Forward pass complete" << std::endl;

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

            output->setBackwardFn([self_input, self_input_cpu, self_Q, self_K, self_V, self_Wq, self_Wk, self_Wv, self_Wo,
                                   self_bq, self_bk, self_bv, self_bo,
                                   self_concat, output, self_num_heads,
                                   self_d_model, seq_len, head_size, causal_mask, scale_factor]() {

                self_Wo->getGrad().add_inplace(self_concat->getData().transpose().matmul(output->getGrad()));

                Tensor db_o(1, self_d_model, Device::CPU);
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

                Tensor dQ(seq_len, self_d_model, Device::CPU);
                Tensor dK(seq_len, self_d_model, Device::CPU);
                Tensor dV(seq_len, self_d_model, Device::CPU);
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
                Tensor dAttended(seq_len, head_size, Device::CPU);
                Tensor scores(seq_len, seq_len);
                Tensor dScores(seq_len, seq_len, Device::CPU);

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

                self_Wq->getGrad().add_inplace(self_input_cpu->getData().transpose().matmul(dQ));
                self_Wk->getGrad().add_inplace(self_input_cpu->getData().transpose().matmul(dK));
                self_Wv->getGrad().add_inplace(self_input_cpu->getData().transpose().matmul(dV));

                Tensor db_q(1, self_d_model, Device::CPU);
                Tensor db_k(1, self_d_model, Device::CPU);
                Tensor db_v(1, self_d_model, Device::CPU);
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
        // 3D case
        std::cerr << "[DEBUG] MHA: Entering 3D case" << std::endl;
        int batch_size = input_tensor.getBatchSize();
        int seq_len = input_tensor.getRows();
        int head_size = d_model / num_heads;

        std::cerr << "[DEBUG] MHA: Computing Q, K, V projections" << std::endl;
        auto Q = input->matmul(W_q)->add(b_q);
        auto K = input->matmul(W_k)->add(b_k);
        auto V = input->matmul(W_v)->add(b_v);

        std::cerr << "[DEBUG] MHA: Q device=" << (Q->getData().getDevice() == Device::CUDA ? "CUDA" : "CPU") << std::endl;
        std::cerr << "[DEBUG] MHA: Transferring Q, K, V to CPU" << std::endl;
        Tensor Q_cpu = (Q->getData().getDevice() == Device::CUDA) ? Q->getData().to(Device::CPU) : Q->getData();
        Tensor K_cpu = (K->getData().getDevice() == Device::CUDA) ? K->getData().to(Device::CPU) : K->getData();
        Tensor V_cpu = (V->getData().getDevice() == Device::CUDA) ? V->getData().to(Device::CPU) : V->getData();
        std::cerr << "[DEBUG] MHA: Transfer complete, Q_cpu device=" << (Q_cpu.getDevice() == Device::CUDA ? "CUDA" : "CPU") << std::endl;

        Tensor result(batch_size, seq_len, d_model, Device::CPU);
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

        const float* Q_data = Q_cpu.raw();
        const float* K_data = K_cpu.raw();
        const float* V_data = V_cpu.raw();
        float* result_data = result.raw();

        Tensor causal_mask = Tensor::create_causal_mask(seq_len);
        const float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_size));
        const float* mask_data = causal_mask.raw();

        Tensor Q_head(seq_len, head_size, Device::CPU);
        Tensor K_head(seq_len, head_size, Device::CPU);
        Tensor V_head(seq_len, head_size, Device::CPU);
        Tensor scores(seq_len, seq_len, Device::CPU);

        std::cerr << "[DEBUG] MHA: Getting raw pointers" << std::endl;
        float* Q_head_data = Q_head.raw();
        float* K_head_data = K_head.raw();
        float* V_head_data = V_head.raw();
        float* scores_data = scores.raw();

        std::cerr << "[DEBUG] MHA: Starting attention computation for " << batch_size << " batches, " << num_heads << " heads" << std::endl;
        for (int b = 0; b < batch_size; b++) {
            const int batch_offset = b * seq_len * d_model;

            for (int h = 0; h < num_heads; h++) {
                const int head_offset = h * head_size;

                for (int i = 0; i < seq_len; i++) {
                    const float* q_row = Q_data + batch_offset + i * d_model + head_offset;
                    const float* k_row = K_data + batch_offset + i * d_model + head_offset;
                    const float* v_row = V_data + batch_offset + i * d_model + head_offset;

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

                Tensor attended(seq_len, head_size, Device::CPU);
                blas_sgemm(attention_weights.raw(), V_head_data, attended.raw(),
                          seq_len, head_size, seq_len, false, false);

                const float* attended_data = attended.raw();
                for (int i = 0; i < seq_len; i++) {
                    float* out_row = result_data + batch_offset + i * d_model + head_offset;
                    const float* in_row = attended_data + i * head_size;
                    std::memcpy(out_row, in_row, head_size * sizeof(float));
                }
            }
        }

        std::cerr << "[DEBUG] MHA: Attention computation complete" << std::endl;
        std::cerr << "[DEBUG] MHA: Transferring result back to " << (input_tensor.getDevice() == Device::CUDA ? "CUDA" : "CPU") << std::endl;
        Tensor result_device = (input_tensor.getDevice() == Device::CUDA) ? result.to(Device::CUDA) : result;
        std::cerr << "[DEBUG] MHA: Creating output variable" << std::endl;
        auto concat_var = Variable::create(result_device, input->requiresGrad());
        auto self_concat = concat_var;
        std::cerr << "[DEBUG] MHA: Computing final linear projection" << std::endl;
        auto output = concat_var->matmul(W_o)->add(b_o);

        if (training && dropout_rate > 0.0f) {
            std::cerr << "[DEBUG] MHA: Applying dropout" << std::endl;
            output = output->dropout(dropout_rate, training);
        }
        std::cerr << "[DEBUG] MHA: Forward pass complete" << std::endl;

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
            size_t self_batch_size = batch_size;

            output->setBackwardFn([self_input, self_Q, self_K, self_V, self_Wq, self_Wk, self_Wv, self_Wo,
                                   self_bq, self_bk, self_bv, self_bo,
                                   self_concat, output, self_num_heads,
                                   self_d_model, self_batch_size, seq_len, head_size, causal_mask, scale_factor]() {

                const float* concat_data = self_concat->getData().raw();
                const float* output_grad_data = output->getGrad().raw();
                const int slice_size = seq_len * self_d_model;

                Tensor concat_slice(seq_len, self_d_model, Device::CPU);
                Tensor output_grad_slice(seq_len, self_d_model, Device::CPU);

                for (size_t b = 0; b < self_batch_size; b++) {
                    std::memcpy(concat_slice.raw(), concat_data + b * slice_size, slice_size * sizeof(float));
                    std::memcpy(output_grad_slice.raw(), output_grad_data + b * slice_size, slice_size * sizeof(float));
                    self_Wo->getGrad().add_inplace(concat_slice.transpose().matmul(output_grad_slice));
                }

                Tensor db_o(1, self_d_model, Device::CPU);
                db_o.fill(0.0f);
                float* db_o_data = db_o.raw();

                for (size_t b = 0; b < self_batch_size; b++) {
                    for (int i = 0; i < seq_len; i++) {
                        for (int j = 0; j < self_d_model; j++) {
                            db_o_data[j] += output_grad_data[b * slice_size + i * self_d_model + j];
                        }
                    }
                }
                self_bo->getGrad().add_inplace(db_o);

                Tensor dConcat = output->getGrad().matmul(self_Wo->getData().transpose());

                Tensor dQ(self_batch_size, seq_len, self_d_model, Device::CPU);
                Tensor dK(self_batch_size, seq_len, self_d_model, Device::CPU);
                Tensor dV(self_batch_size, seq_len, self_d_model, Device::CPU);
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
                Tensor dAttended(seq_len, head_size, Device::CPU);
                Tensor scores(seq_len, seq_len);
                Tensor dScores(seq_len, seq_len, Device::CPU);

                float* Q_head_data = Q_head.raw();
                float* K_head_data = K_head.raw();
                float* V_head_data = V_head.raw();
                float* dAttended_data = dAttended.raw();
                float* scores_data = scores.raw();
                float* dScores_data = dScores.raw();

                for (size_t b = 0; b < self_batch_size; b++) {
                    const int batch_offset = b * seq_len * self_d_model;

                    for (int h = 0; h < self_num_heads; h++) {
                        const int start_col = h * head_size;

                        for (int i = 0; i < seq_len; i++) {
                            std::memcpy(dAttended_data + i * head_size,
                                       dConcat_data + batch_offset + i * self_d_model + start_col,
                                       head_size * sizeof(float));
                            std::memcpy(Q_head_data + i * head_size,
                                       Q_data + batch_offset + i * self_d_model + start_col,
                                       head_size * sizeof(float));
                            std::memcpy(K_head_data + i * head_size,
                                       K_data + batch_offset + i * self_d_model + start_col,
                                       head_size * sizeof(float));
                            std::memcpy(V_head_data + i * head_size,
                                       V_data + batch_offset + i * self_d_model + start_col,
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
                                dQ_data[batch_offset + i * self_d_model + start_col + j] += dQ_head_data[i * head_size + j];
                                dK_data[batch_offset + i * self_d_model + start_col + j] += dK_head_data[i * head_size + j];
                                dV_data[batch_offset + i * self_d_model + start_col + j] += dV_head_data[i * head_size + j];
                            }
                        }
                    }
                }

                const float* input_data = self_input->getData().raw();
                Tensor input_slice(seq_len, self_d_model, Device::CPU);
                Tensor dQ_slice(seq_len, self_d_model, Device::CPU);
                Tensor dK_slice(seq_len, self_d_model, Device::CPU);
                Tensor dV_slice(seq_len, self_d_model, Device::CPU);

                for (size_t b = 0; b < self_batch_size; b++) {
                    std::memcpy(input_slice.raw(), input_data + b * slice_size, slice_size * sizeof(float));
                    std::memcpy(dQ_slice.raw(), dQ_data + b * slice_size, slice_size * sizeof(float));
                    std::memcpy(dK_slice.raw(), dK_data + b * slice_size, slice_size * sizeof(float));
                    std::memcpy(dV_slice.raw(), dV_data + b * slice_size, slice_size * sizeof(float));

                    Tensor input_T = input_slice.transpose();
                    self_Wq->getGrad().add_inplace(input_T.matmul(dQ_slice));
                    self_Wk->getGrad().add_inplace(input_T.matmul(dK_slice));
                    self_Wv->getGrad().add_inplace(input_T.matmul(dV_slice));
                }

                Tensor db_q(1, self_d_model, Device::CPU);
                Tensor db_k(1, self_d_model, Device::CPU);
                Tensor db_v(1, self_d_model, Device::CPU);
                db_q.fill(0.0f);
                db_k.fill(0.0f);
                db_v.fill(0.0f);

                float* db_q_data = db_q.raw();
                float* db_k_data = db_k.raw();
                float* db_v_data = db_v.raw();

                for (size_t b = 0; b < self_batch_size; b++) {
                    for (int i = 0; i < seq_len; i++) {
                        for (int j = 0; j < self_d_model; j++) {
                            const int idx = b * slice_size + i * self_d_model + j;
                            db_q_data[j] += dQ_data[idx];
                            db_k_data[j] += dK_data[idx];
                            db_v_data[j] += dV_data[idx];
                        }
                    }
                }

                self_bq->getGrad().add_inplace(db_q);
                self_bk->getGrad().add_inplace(db_k);
                self_bv->getGrad().add_inplace(db_v);
                Tensor dInput(self_batch_size, seq_len, self_d_model, Device::CPU);
                float* dInput_data = dInput.raw();

                for (size_t b = 0; b < self_batch_size; b++) {
                    std::memcpy(dQ_slice.raw(), dQ_data + b * slice_size, slice_size * sizeof(float));
                    std::memcpy(dK_slice.raw(), dK_data + b * slice_size, slice_size * sizeof(float));
                    std::memcpy(dV_slice.raw(), dV_data + b * slice_size, slice_size * sizeof(float));

                    Tensor dInput_slice = dQ_slice.matmul(self_Wq->getData().transpose())
                                         .add(dK_slice.matmul(self_Wk->getData().transpose()))
                                         .add(dV_slice.matmul(self_Wv->getData().transpose()));

                    std::memcpy(dInput_data + b * slice_size, dInput_slice.raw(), slice_size * sizeof(float));
                }
                self_input->getGrad().add_inplace(dInput);
            });
        }
        return output;
    }
}