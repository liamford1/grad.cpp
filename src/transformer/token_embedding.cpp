#include "transformer/tensor.h"
#include "transformer/token_embedding.h"
#include <iostream>
#include <stdexcept>

TokenEmbedding::TokenEmbedding(int vocab_size, int d_model) :
    vocab_size(vocab_size),
    d_model(d_model),
    embedding_table(vocab_size, d_model) 
{
    embedding_table.xavier(vocab_size, d_model);
}

TokenEmbedding::~TokenEmbedding() {}

std::shared_ptr<Variable> TokenEmbedding::forward(std::shared_ptr<Variable> input_ids) const {
    Tensor input_tensor = input_ids->getData();

    if (input_tensor.getIs3D()) {
        throw std::invalid_argument("input_ids shoudl be 2D tensor not 3D");
    }

    if(input_tensor.getCols() == 1) {
        int seq_len = input_tensor.getRows();

        Tensor result(seq_len, d_model);

        for (int i = 0; i < seq_len; i++) {
            int token_id = static_cast<int>(input_tensor.getValue(i, 0));

            if (token_id < 0 || token_id >= vocab_size) {
                throw std::out_of_range("Token ID out of vocabulary range");
            }

            for (int j = 0; j < d_model; j++) {
                result.setValue(i, j, embedding_table.getValue(token_id, j));
            }
        }
        return Variable::create(result, input_ids->requiresGrad());
    } else {
        int batch_size = input_tensor.getRows();
        int seq_len = input_tensor.getCols();

        Tensor result(batch_size, seq_len, d_model);
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < seq_len; i++) {
                int token_id = static_cast<int>(input_tensor.getValue(b, i));

                if(token_id < 0 || token_id >= vocab_size) {
                    throw std::out_of_range("Token Id out of vocab range");
                }
                for (int j = 0; j < d_model; j++) {
                    result.setValue(b, i, j, embedding_table.getValue(token_id, j));
                }
            }
        }
        return Variable::create(result, input_ids->requiresGrad());
    }
}

