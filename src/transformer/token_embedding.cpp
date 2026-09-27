#include "grad/transformer/token_embedding.h"
#include "grad/transformer/device.h"
#include "grad/transformer/metal_ops.h"
#include "grad/utils/narrow.h"
#include "metal_graph.h"
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace grad {

namespace {

// A uint32 index array in (float-typed) Metal tensor storage. The tensor
// is fresh and never yet given to the GPU, so the write does not wait.
Tensor index_tensor(const std::vector<uint32_t>& v) {
    Tensor t = Tensor::uninitialized(Shape{v.empty() ? size_t{1} : v.size()});
    std::memcpy(t.raw(), v.data(), v.size() * sizeof(uint32_t));
    return t;
}

// Metal mode. The ids are read on the CPU (a batch fresh from the loader,
// so no wait) to validate them, which is also the moment to sort the
// positions by token for the backward pass: a stable counting sort gives
// each distinct token its positions in increasing order, the order the CPU
// loop accumulates them in, and the GPU sums each token's segment with one
// thread per column and no atomics.
std::shared_ptr<Variable> embedding_metal(const std::shared_ptr<Variable>& input_ids,
                                          const std::shared_ptr<Variable>& table, int vocab_size,
                                          Tensor&& result, size_t total, size_t d, float scale) {
    const Tensor& ids = input_ids->getData();
    const float* id_values = ids.raw();
    const auto vocab = static_cast<size_t>(vocab_size);
    for (size_t t = 0; t < total; t++) {
        const int token_id = static_cast<int>(id_values[t]);
        if (token_id < 0 || token_id >= vocab_size) {
            throw std::out_of_range("Token ID out of vocab range");
        }
    }
    metal::ops::embedding(ids.device_data(), table->getData().device_data(), result.device_data(),
                          total, d, scale);

    const bool needs_grad = compute_requires_grad(table, input_ids);
    auto output = Variable::create(std::move(result), needs_grad);
    if (!needs_grad || !table->requiresGrad()) return output;

    std::vector<uint32_t> count(vocab, 0);
    for (size_t t = 0; t < total; t++) count[static_cast<size_t>(id_values[t])]++;
    std::vector<uint32_t> tokens;
    std::vector<uint32_t> starts{0};
    std::vector<uint32_t> next(vocab, 0);
    for (size_t v = 0; v < vocab; v++) {
        if (count[v] == 0) continue;
        next[v] = starts.back();
        tokens.push_back(static_cast<uint32_t>(v));
        starts.push_back(starts.back() + count[v]);
    }
    std::vector<uint32_t> order(total);
    for (size_t t = 0; t < total; t++) {
        order[next[static_cast<size_t>(id_values[t])]++] = static_cast<uint32_t>(t);
    }
    auto seg_tokens = std::make_shared<Tensor>(index_tensor(tokens));
    auto seg_starts = std::make_shared<Tensor>(index_tensor(starts));
    auto seg_order = std::make_shared<Tensor>(index_tensor(order));
    const size_t unique = tokens.size();

    output->setBackward(
        {table}, [table, seg_tokens, seg_starts, seg_order, unique, d, scale](Variable& node) {
        metal::ops::embedding_backward(
            seg_tokens->device_data(), seg_starts->device_data(), seg_order->device_data(), unique,
            node.getGrad().device_data(), metal_graph::grad_for_write(*table), d, scale);
    });
    return output;
}

}  // namespace

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
    if (metal_mode()) {
        return embedding_metal(input_ids, embedding_table, vocab_size_, std::move(result), total, d,
                               embedding_scale);
    }

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
