#pragma once

#include "tensor.h"
#include "variable.h"
#include <memory>

class PositionalEncoding {
    private:
        int max_len;
        int d_model;
        Tensor encoding_table;
        void computeEncodings();
    public:
        PositionalEncoding(int max_len, int d_model);
        ~PositionalEncoding();

        std::shared_ptr<Variable> forward(std::shared_ptr<Variable> embeddings) const;

        int getMaxLen() const { return max_len; }
        int getDModel() const { return d_model; }
};