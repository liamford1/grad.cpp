#pragma once
#include "tensor.h"
#include "variable.h"
#include <memory>

class TokenEmbedding {
    private:
        int vocab_size;
        int d_model;
        Tensor embedding_table;
    public:
        TokenEmbedding(int vocab_size, int d_model);
        ~TokenEmbedding();
        std::shared_ptr<Variable> forward(std::shared_ptr<Variable> input_ids) const;

        int getVocabSize() const { return vocab_size; }
        int getDModel() const { return d_model; }

        const Tensor& getEmbeddingTable() const { return embedding_table; }

        void setEmbeddingTable(const Tensor& new_embedding_table) {
            embedding_table = new_embedding_table;
        }
};