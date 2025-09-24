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
};

#endif