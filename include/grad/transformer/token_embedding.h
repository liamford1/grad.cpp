#pragma once
#include "grad/transformer/tensor.h"
#include "grad/transformer/variable.h"
#include <memory>

namespace grad {

class TokenEmbedding {
private:
    int vocab_size_;
    int d_model_;
    float embedding_scale;
    std::shared_ptr<Variable> embedding_table;

public:
    TokenEmbedding(int vocab_size, int d_model);
    std::shared_ptr<Variable> forward(std::shared_ptr<Variable> input_ids) const;

    int getVocabSize() const { return vocab_size_; }
    int getDModel() const { return d_model_; }

    std::shared_ptr<Variable> getEmbeddingTable() const { return embedding_table; }
    float getScale() const { return embedding_scale; }

    std::vector<std::shared_ptr<Variable>> parameters() const { return {embedding_table}; }

    void setEmbeddingTable(const Tensor& new_embedding_table) {
        embedding_table = Variable::create(new_embedding_table, true);
    }
};

}  // namespace grad
