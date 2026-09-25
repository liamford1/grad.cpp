#include "grad/transformer/positional_encoding.h"
#include <stdexcept>

namespace grad {

PositionalEncoding::PositionalEncoding(int max_len, int d_model) : 
    max_len(max_len),
    d_model(d_model)
{
    Tensor pos_emb(max_len, d_model);
    pos_emb.xavier(max_len, d_model);
    position_embeddings = Variable::create(pos_emb, true);
}

// Adds the first seq_len rows of the position table to (seq, d) or
// (batch, seq, d) embeddings, broadcasting over the batch.
std::shared_ptr<Variable> PositionalEncoding::forward(std::shared_ptr<Variable> embeddings) const {
    const Tensor& emb_tensor = embeddings->getData();
    const int seq_len = static_cast<int>(emb_tensor.getRows());
    if (seq_len > max_len) {
        throw std::out_of_range("Sequence length exceeds max_len");
    }

    const Tensor pos_slice = position_embeddings->getData().slice(0, seq_len, 0, d_model);
    const bool needs_grad = compute_requires_grad(embeddings, position_embeddings);
    auto output = Variable::create(emb_tensor.add(pos_slice), needs_grad);

    if (needs_grad) {
        auto self_pos_emb = position_embeddings;
        output->setBackward({embeddings, position_embeddings},
                            [embeddings, self_pos_emb](Variable& output) {
            const Tensor& dOut = output.getGrad();

            if (embeddings->requiresGrad()) {
                embeddings->ensureGrad();
                embeddings->getGrad().add_inplace(dOut);
            }

            // Row i of the table receives the sum over the batch of every
            // sequence's position-i gradient, accumulated batch by batch.
            if (self_pos_emb->requiresGrad()) {
                self_pos_emb->ensureGrad();
                const size_t rows = dOut.getRows();
                const size_t cols = dOut.getCols();
                const float* g = dOut.raw();
                float* dPos = self_pos_emb->getGrad().raw();
                for (size_t b = 0; b < dOut.getBatchSize(); b++) {
                    for (size_t i = 0; i < rows; i++) {
                        const float* g_row = g + (b * rows + i) * cols;
                        float* d_row = dPos + i * cols;
                        for (size_t j = 0; j < cols; j++) {
                            d_row[j] += g_row[j];
                        }
                    }
                }
            }
        });
    }
    return output;
}

}  // namespace grad
