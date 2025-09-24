#include "tensor.h"
#include "token_embedding.h"
#include <iostream>

TokenEmbedding::TokenEmbedding(int vocab_size, int d_model) :
    vocab_size(vocab_size),
    d_model(d_model),
    embedding_table(vocab_size, d_model) 
{
    embedding_table.xavier(vocab_size, d_model);
}

TokenEmbedding::~TokenEmbedding() {}

Tensor TokenEmbedding::forward(const Tensor& input_ids) const {
    int seq_len = input_ids.getRows();

    if (input_ids.getCols() != 1) {
        throw std::invalid_argument("input_ids should have shape [seq_len, 1]");
    }

    Tensor result(seq_len, d_model);

    for (int i = 0; i < seq_len; i++) {
        int token_id = static_cast<int>(input_ids.getValue(i, 0));

        if (token_id < 0 || token_id >= vocab_size) {
            throw std::out_of_range("Token ID out of vocabulary range");
        }

        for (int j = 0; j < d_model; j++) {
            result.setValue(i, j, embedding_table.getValue(token_id, j));
        }
    }

    return result;
}

