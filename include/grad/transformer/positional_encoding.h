#pragma once

#include "grad/transformer/tensor.h"
#include "grad/transformer/variable.h"
#include <memory>

namespace grad {

class PositionalEncoding {
private:
    int max_len_;
    int d_model_;
    std::shared_ptr<Variable> position_embeddings;

public:
    PositionalEncoding(int max_len, int d_model);

    std::shared_ptr<Variable> forward(std::shared_ptr<Variable> embeddings) const;

    int getMaxLen() const { return max_len_; }
    int getDModel() const { return d_model_; }
    std::shared_ptr<Variable> getPositionEmbeddings() const { return position_embeddings; }
    void setPositionEmbeddings(const Tensor& new_embeddings) {
        position_embeddings = Variable::create(new_embeddings, true);
    }
};

}  // namespace grad
