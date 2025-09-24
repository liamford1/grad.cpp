#ifndef POSITIONAL_ENCODING_H
#define POSITIONAL_ENCODING_H

#include "tensor.h"

class PositionalEncoding {
    private:
        int max_len;
        int d_model;
        Tensor encoding_table;
        void computeEncodings();
    public:
        PositionalEncoding(int max_len, int d_model);
        ~PositionalEncoding();

        Tensor forward(const Tensor& embeddings) const;

        int getMaxLen() const { return max_len; }
        int getDModel() const { return d_model; }
};

#endif