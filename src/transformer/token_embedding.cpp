#include "grad/transformer/token_embedding.h"
#include "grad/utils/narrow.h"
#include <stdexcept>
#include <vector>

namespace grad {

TokenEmbedding::TokenEmbedding(int vocab_size, int d_model)
    : vocab_size_(vocab_size), d_model_(d_model) {
    const size_t rows = narrow<size_t>(vocab_size);
    const size_t cols = narrow<size_t>(d_model);
    Tensor table(rows, cols);
    table.xavier(rows, cols);
    embedding_table = Variable::create(table, true);
}

// Accepts token IDs in any of three layouts, all contiguous row-major:
//   (batch, seq, 1) 3D -> (batch, seq, d_model)
//   (seq, 1)        2D -> (seq, d_model)
//   (batch, seq)    2D -> (batch, seq, d_model)
std::shared_ptr<Variable> TokenEmbedding::forward(
    const std::shared_ptr<Variable>& input_ids) const {
    const Tensor& input_tensor = input_ids->getData();

    bool output_3d;
    size_t batch_size, seq_len;
    if (input_tensor.getIs3D()) {
        if (input_tensor.getCols() != 1) {
            throw std::invalid_argument("3D input_ids must have shape (batch, seq_len, 1)");
        }
        batch_size = input_tensor.getBatchSize();
        seq_len = input_tensor.getRows();
        output_3d = true;
    } else if (input_tensor.getCols() == 1) {
        batch_size = 1;
        seq_len = input_tensor.getRows();
        output_3d = false;
    } else {
        batch_size = input_tensor.getRows();
        seq_len = input_tensor.getCols();
        output_3d = true;
    }

    const size_t total = batch_size * seq_len;
    const size_t d = static_cast<size_t>(d_model_);
    // Every row is written below, so no zero-fill.
    Tensor result =
        Tensor::uninitialized(output_3d ? Shape{batch_size, seq_len, d} : Shape{seq_len, d});

    // Validated ids as table row indices, kept for the backward pass.
    std::vector<size_t> token_rows(total);
    const float* ids = input_tensor.raw();
    const float* table = embedding_table->getData().raw();
    float* out = result.raw();

    for (size_t t = 0; t < total; t++) {
        const int token_id = static_cast<int>(ids[t]);
        if (token_id < 0 || token_id >= vocab_size_) {
            throw std::out_of_range("Token ID out of vocab range");
        }
        const size_t token_row = static_cast<size_t>(token_id);
        token_rows[t] = token_row;

        const float* row = table + token_row * d;
        float* dst = out + t * d;
        for (size_t j = 0; j < d; j++) {
            dst[j] = row[j] * embedding_scale;
        }
    }

    // The lookup differentiates w.r.t. the embedding table, not the discrete
    // token IDs, so grad tracking must key off the table.
    const bool needs_grad = compute_requires_grad(embedding_table, input_ids);
    auto output = Variable::create(std::move(result), needs_grad);

    if (needs_grad && embedding_table->requiresGrad()) {
        auto table_var = embedding_table;
        const float scale = embedding_scale;

        output->setBackward({embedding_table},
                            [table_var, rows = std::move(token_rows), scale, d](Variable& node) {
            table_var->ensureGrad();
            const float* dOut = node.getGrad().raw();
            float* dTable = table_var->getGrad().raw();

            for (size_t t = 0; t < rows.size(); t++) {
                const float* src = dOut + t * d;
                float* dst = dTable + rows[t] * d;
                for (size_t j = 0; j < d; j++) {
                    dst[j] += src[j] * scale;
                }
            }
        });
    }
    return output;
}

}  // namespace grad
