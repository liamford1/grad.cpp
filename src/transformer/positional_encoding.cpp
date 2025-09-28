#include "transformer/tensor.h"
#include "transformer/positional_encoding.h"
#include <iostream>
#include <cmath>
#include <stdexcept>

PositionalEncoding::PositionalEncoding(int max_len, int d_model) : 
    max_len(max_len),
    d_model(d_model),
    encoding_table(max_len, d_model)
{
    computeEncodings();
}

PositionalEncoding::~PositionalEncoding() {}

void PositionalEncoding::computeEncodings() {
    for (int i = 0; i < max_len; i++) {
        for (int j = 0; j < d_model; j++) {
            float angle = i / pow(10000.0f, (2.0f * (j / 2)) / d_model);

            if (j % 2 == 0) {
                encoding_table.setValue(i, j, sin(angle));
            } else {
                encoding_table.setValue(i, j, cos(angle));
            }
        }
    }
}

Tensor PositionalEncoding::forward(const Tensor& embeddings) const {
    if (!embeddings.getIs3D()) {
        int seq_len = embeddings.getRows();
        if (seq_len > max_len) {
            throw std::out_of_range("Sequence length exceeds max sequence length");
        }
        if (embeddings.getCols() != d_model) {
            throw std::invalid_argument("Embedding dimensions do not match");
        }

        Tensor result(seq_len, d_model);
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < d_model; j++) {
                float embedding_val = embeddings.getValue(i, j);
                float pos_encoding = encoding_table.getValue(i, j);
                result.setValue(i, j, embedding_val + pos_encoding);
            }
        }
        return result;
    } else {
        int batch_size = embeddings.getBatchSize();
        int seq_len = embeddings.getRows();
        
        if (seq_len > max_len) {
            throw std::out_of_range("Sequence length exceeds max sequence length");
        }
        if (embeddings.getCols() != d_model) {
            throw std::invalid_argument("Embedding dimensions do not match");
        }

        Tensor result(batch_size, seq_len, d_model);
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < d_model; j++) {
                    float embedding_val = embeddings.getValue(b, i, j);
                    float pos_encoding = encoding_table.getValue(i, j);
                    result.setValue(b, i, j, embedding_val + pos_encoding);
                }
            }
        }
        return result;
    }
}