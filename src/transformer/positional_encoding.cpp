#include "grad/transformer/positional_encoding.h"
#include "grad/transformer/device.h"
#include "grad/transformer/metal_ops.h"
#include "grad/utils/narrow.h"
#include "metal_graph.h"
#include <stdexcept>

namespace grad {

PositionalEncoding::PositionalEncoding(int max_len, int d_model)
    : max_len_(max_len), d_model_(d_model) {
    const size_t rows = narrow<size_t>(max_len);
    const size_t cols = narrow<size_t>(d_model);
    Tensor pos_emb(rows, cols);
    pos_emb.xavier(rows, cols);
    position_embeddings = Variable::create(pos_emb, true);
}

// Adds the first seq_len rows of the position table to (seq, d) or
// (batch, seq, d) embeddings, broadcasting over the batch.
std::shared_ptr<Variable> PositionalEncoding::forward(
    const std::shared_ptr<Variable>& embeddings) const {
    const Tensor& emb_tensor = embeddings->getData();
    const size_t seq_len = emb_tensor.getRows();
    if (seq_len > static_cast<size_t>(max_len_)) {
        throw std::out_of_range("Sequence length exceeds max_len");
    }
    if (metal_mode()) {
        // The first seq_len table rows, added to every sequence; backward
        // sums the batch's gradients into those rows (a column sum over the
        // batch of (seq_len * d)-wide rows).
        const size_t d = emb_tensor.getCols();
        Tensor out = Tensor::empty_like(emb_tensor);
        metal::ops::add_rows(emb_tensor.device_data(), position_embeddings->getData().device_data(),
                             out.device_data(), emb_tensor.numel(), d, seq_len);
        const bool needs_grad = compute_requires_grad(embeddings, position_embeddings);
        auto output = Variable::create(std::move(out), needs_grad);
        if (needs_grad) {
            auto table = position_embeddings;
            output->setBackward({embeddings, position_embeddings},
                                [embeddings, table, seq_len, d](Variable& node) {
                const Tensor& dOut = node.getGrad();
                if (embeddings->requiresGrad()) {
                    metal::ops::accumulate(metal_graph::grad_for_write(*embeddings),
                                           dOut.device_data(), dOut.numel());
                }
                if (table->requiresGrad()) {
                    metal::ops::column_sums(dOut.device_data(), dOut.getBatchSize(), seq_len * d,
                                            metal_graph::grad_for_write(*table));
                }
            });
        }
        return output;
    }

    const Tensor pos_slice =
        position_embeddings->getData().slice(0, seq_len, 0, static_cast<size_t>(d_model_));
    const bool needs_grad = compute_requires_grad(embeddings, position_embeddings);
    auto output = Variable::create(emb_tensor.add(pos_slice), needs_grad);

    if (needs_grad) {
        auto self_pos_emb = position_embeddings;
        output->setBackward({embeddings, position_embeddings},
                            [embeddings, self_pos_emb](Variable& node) {
            const Tensor& dOut = node.getGrad();

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
