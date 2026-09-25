#include "grad/transformer/token_embedding.h"
#include <stdexcept>
#include <vector>

namespace grad {

TokenEmbedding::TokenEmbedding(int vocab_size, int d_model) :
    vocab_size(vocab_size),
    d_model(d_model),
    embedding_scale(1.0f)
{
    Tensor table(vocab_size, d_model);
    table.xavier(vocab_size, d_model);
    embedding_table = Variable::create(table, true);
}


// Accepts token IDs in any of three layouts, all contiguous row-major:
//   (batch, seq, 1) 3D -> (batch, seq, d_model)
//   (seq, 1)        2D -> (seq, d_model)
//   (batch, seq)    2D -> (batch, seq, d_model)
std::shared_ptr<Variable> TokenEmbedding::forward(std::shared_ptr<Variable> input_ids) const {
    const Tensor& input_tensor = input_ids->getData();

    bool output_3d;
    int batch_size, seq_len;
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

    const int total = batch_size * seq_len;
    // Every row is written below, so no zero-fill.
    Tensor result = Tensor::uninitialized(
        output_3d ? Shape{static_cast<size_t>(batch_size), static_cast<size_t>(seq_len),
                          static_cast<size_t>(d_model)}
                  : Shape{static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});

    std::vector<int> token_ids(total);
    const float* ids = input_tensor.raw();
    const float* table = embedding_table->getData().raw();
    float* out = result.raw();

    for (int t = 0; t < total; t++) {
        int token_id = static_cast<int>(ids[t]);
        if (token_id < 0 || token_id >= vocab_size) {
            throw std::out_of_range("Token ID out of vocab range");
        }
        token_ids[t] = token_id;

        const float* row = table + static_cast<size_t>(token_id) * d_model;
        float* dst = out + static_cast<size_t>(t) * d_model;
        for (int j = 0; j < d_model; j++) {
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
        const int dm = d_model;

        output->setBackward({embedding_table},
                            [table_var, ids = std::move(token_ids), scale, dm](Variable& output) {
            table_var->ensureGrad();
            const float* dOut = output.getGrad().raw();
            float* dTable = table_var->getGrad().raw();

            for (size_t t = 0; t < ids.size(); t++) {
                const float* src = dOut + t * dm;
                float* dst = dTable + static_cast<size_t>(ids[t]) * dm;
                for (int j = 0; j < dm; j++) {
                    dst[j] += src[j] * scale;
                }
            }
        });
    }
    return output;
}

}  // namespace grad
