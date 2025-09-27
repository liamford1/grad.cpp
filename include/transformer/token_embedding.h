#ifndef TOKEN_EMBEDDING_H
#define TOKEN_EMBEDDING_H

#include "tensor.h"

class TokenEmbedding {
    private:
        int vocab_size;
        int d_model;
        Tensor embedding_table;
    public:
        TokenEmbedding(int vocab_size, int d_model);
        ~TokenEmbedding();
        Tensor forward(const Tensor& input_ids) const;

        int getVocabSize() const { return vocab_size; }
        int getDModel() const { return d_model; }

        const Tensor& getEmbeddingTable() const { return embedding_table; }

        void setEmbeddingTable(const Tensor& new_embedding_table) {
            embedding_table = new_embedding_table;
        }
};

#endif