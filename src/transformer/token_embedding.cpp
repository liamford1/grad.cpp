#include "transformer/tensor.h"
#include "transformer/token_embedding.h"
#include <iostream>
#include <stdexcept>

TokenEmbedding::TokenEmbedding(int vocab_size, int d_model) :
    vocab_size(vocab_size),
    d_model(d_model)
{
    Tensor embedding_tensor(vocab_size, d_model);
    embedding_tensor.xavier(vocab_size, d_model);
    embedding_table = Variable::create(embedding_tensor, true);
}

TokenEmbedding::~TokenEmbedding() {}

std::shared_ptr<Variable> TokenEmbedding::forward(std::shared_ptr<Variable> input_ids) const {
    const Tensor& input_tensor = input_ids->getData();

    if (input_tensor.getIs3D()) {
        throw std::invalid_argument("input_ids should be 2D tensor not 3D");
    }

    if (input_tensor.getCols() == 1) {
        int seq_len = input_tensor.getRows();
        Tensor result(seq_len, d_model);

        std::vector<int> token_ids(seq_len);
        for (int i = 0; i < seq_len; i++) {
            int token_id = static_cast<int>(input_tensor.getValue(i, 0));
            if (token_id < 0 || token_id >= vocab_size) {
                throw std::out_of_range("Token ID out of vocabulary range");
            }
            token_ids[i] = token_id;
            
            for (int j = 0; j < d_model; j++) {
                result.setValue(i, j, embedding_table->getData().getValue(token_id, j));
            }
        }

        auto output = Variable::create(result, input_ids->requiresGrad());

        if (input_ids->requiresGrad()) {
            auto self_embedding = embedding_table;
            int self_vocab_size = vocab_size;
            int self_d_model = d_model;

            output->addChild(input_ids);
            output->addChild(embedding_table);

            output->setBackwardFn([self_embedding, output, token_ids, self_vocab_size, 
                                   self_d_model, seq_len]() {
                
                Tensor dEmbedding(self_vocab_size, self_d_model);
                dEmbedding.fill(0.0f);

                for (int i = 0; i < seq_len; i++) {
                    int token_id = token_ids[i];
                    for (int j = 0; j < self_d_model; j++) {
                        float grad = output->getGrad().getValue(i, j);
                        dEmbedding.setValue(token_id, j, 
                            dEmbedding.getValue(token_id, j) + grad);
                    }
                }

                self_embedding->getGrad() = self_embedding->getGrad().add(dEmbedding);
            });
        }
        return output;

    } else {
        int batch_size = input_tensor.getRows();
        int seq_len = input_tensor.getCols();

        Tensor result(batch_size, seq_len, d_model);
        
        std::vector<std::vector<int>> token_ids(batch_size, std::vector<int>(seq_len));
        
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < seq_len; i++) {
                int token_id = static_cast<int>(input_tensor.getValue(b, i));
                if (token_id < 0 || token_id >= vocab_size) {
                    throw std::out_of_range("Token ID out of vocab range");
                }
                token_ids[b][i] = token_id;
                
                for (int j = 0; j < d_model; j++) {
                    result.setValue(b, i, j, embedding_table->getData().getValue(token_id, j));
                }
            }
        }
        auto output = Variable::create(result, input_ids->requiresGrad());

        if (input_ids->requiresGrad()) {
            auto self_embedding = embedding_table;
            int self_vocab_size = vocab_size;
            int self_d_model = d_model;

            output->addChild(input_ids);
            output->addChild(embedding_table);

            output->setBackwardFn([self_embedding, output, token_ids, self_vocab_size, 
                                   self_d_model, batch_size, seq_len]() {
                
                Tensor dEmbedding(self_vocab_size, self_d_model);
                dEmbedding.fill(0.0f);

                for (int b = 0; b < batch_size; b++) {
                    for (int i = 0; i < seq_len; i++) {
                        int token_id = token_ids[b][i];
                        for (int j = 0; j < self_d_model; j++) {
                            float grad = output->getGrad().getValue(b, i, j);
                            dEmbedding.setValue(token_id, j, 
                                dEmbedding.getValue(token_id, j) + grad);
                        }
                    }
                }

                self_embedding->getGrad() = self_embedding->getGrad().add(dEmbedding);
            });
        }
        return output;
    }
}