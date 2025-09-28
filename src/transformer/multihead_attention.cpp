#include "transformer/tensor.h"
#include "transformer/activations.h"
#include "transformer/multihead_attention.h"
#include <iostream>
#include <cmath>
#include <stdexcept>

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads, float dropout_rate) : 
    d_model(d_model),
    num_heads(num_heads),
    dropout_rate(dropout_rate),
    W_q(d_model, d_model),
    W_k(d_model, d_model), 
    W_v(d_model, d_model),
    W_o(d_model, d_model) {
    if (num_heads == 0 || (d_model % num_heads) != 0) {
        throw std::invalid_argument("d_model must be divisible by num_heads and num_heads > 0");
    }
    W_q.xavier(d_model, d_model);
    W_k.xavier(d_model, d_model);
    W_v.xavier(d_model, d_model);
    W_o.xavier(d_model, d_model);
}

MultiHeadAttention::~MultiHeadAttention() {}

std::shared_ptr<Variable> MultiHeadAttention::forward(std::shared_ptr<Variable> input, bool training) const {
    Tensor input_tensor = input->getData();

    if (!input_tensor.getIs3D()) {
        int seq_len = input_tensor.getRows();

        Tensor Q = input_tensor.matmul(W_q);
        Tensor K = input_tensor.matmul(W_k);
        Tensor V = input_tensor.matmul(W_v);

        int head_size = d_model / num_heads;
        Tensor result(input_tensor.getRows(), d_model);

        for (int i = 0; i < num_heads; i++) {
            int start_col = i * head_size;
            
            Tensor Q_head = Q.slice(0, Q.getRows(), start_col, head_size);
            Tensor K_head = K.slice(0, K.getRows(), start_col, head_size);
            Tensor V_head = V.slice(0, V.getRows(), start_col, head_size);
            
            Tensor scores = Q_head.matmul(K_head.transpose());
            scores = scores.scale(1.0f / std::sqrt(head_size));

            Tensor casual_mask = Tensor::create_casual_mask(seq_len);
            scores = scores.add(casual_mask);

            Tensor attention_weights = scores.softmax();
            attention_weights = dropout(attention_weights, dropout_rate, training);
            Tensor attended_values = attention_weights.matmul(V_head);
            
            for (int row = 0; row < input_tensor.getRows(); row++) {
                for (int col = 0; col < head_size; col++) {
                    result.setValue(row, start_col + col, attended_values.getValue(row, col));
                }
            }
        }
        Tensor output = result.matmul(W_o);
        output = dropout(output, dropout_rate, training);
        return Variable::create(output, input->requiresGrad());
    } else {
        int batch_size = input_tensor.getBatchSize();
        int seq_len = input_tensor.getRows();

        Tensor Q = input_tensor.matmul(W_q);
        Tensor K = input_tensor.matmul(W_k);
        Tensor V = input_tensor.matmul(W_v);

        int head_size = d_model / num_heads;
        Tensor result(batch_size, seq_len, d_model);

        for (int i = 0; i < num_heads; i++) {
            int start_col = i * head_size;

            Tensor Q_head(batch_size, seq_len, head_size);
            Tensor K_head(batch_size, seq_len, head_size);
            Tensor V_head(batch_size, seq_len, head_size);

            for (int b = 0; b < batch_size; b++) {
                for (int s = 0; s < seq_len; s++) {
                    for (int h = 0; h < head_size; h++) {
                        Q_head.setValue(b, s, h, Q.getValue(b, s, start_col + h));
                        K_head.setValue(b, s, h, K.getValue(b, s, start_col + h));
                        V_head.setValue(b, s, h, V.getValue(b, s, start_col + h));
                    }
                }
            }

            Tensor K_transposed = K_head.transpose();
            Tensor scores = Q_head.matmul(K_transposed);
            scores = scores.scale(1.0f / std::sqrt(head_size));

            Tensor casual_mask = Tensor::create_casual_mask_batch(batch_size, seq_len);
            scores = scores.add(casual_mask);

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
        Tensor output = result.matmul(W_o);
        output = dropout(output, dropout_rate, training);
        return Variable::create(output, input->requiresGrad());
    }
}