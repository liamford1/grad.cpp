#pragma once
#include "tensor.h"
#include "variable.h"
#include <memory>

class TokenEmbedding {
    private:
        int vocab_size;
        int d_model;
        float embedding_scale;
        std::shared_ptr<Variable> embedding_table;
    public:
        TokenEmbedding(int vocab_size, int d_model);
        ~TokenEmbedding();
        std::shared_ptr<Variable> forward(std::shared_ptr<Variable> input_ids) const;

        int getVocabSize() const { return vocab_size; }
        int getDModel() const { return d_model; }

        std::shared_ptr<Variable> getEmbeddingTable() const { return embedding_table; }

        std::vector<std::shared_ptr<Variable>> parameters() const {
            return {embedding_table};
        }

        void setEmbeddingTable(const Tensor& new_embedding_table) {
            embedding_table = Variable::create(new_embedding_table, true);
        }
};