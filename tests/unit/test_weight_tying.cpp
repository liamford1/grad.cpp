#include <cmath>
#include <iostream>
#include "grad/transformer/gpt_model.h"
#include "grad/transformer/variable.h"
#include "grad/transformer/tensor.h"
#include "../test_util.h"

using namespace grad;

int main() {
    std::cout << "=== Weight Tying Verification ===" << std::endl;
    
    int vocab_size = 256;
    int d_model = 128;
    int num_layers = 2;
    int num_heads = 4;
    int max_len = 512;
    
    GPTModel model(vocab_size, d_model, num_layers, num_heads, max_len, 0.0f);
    
    auto params = model.getAllParameters();
    std::cout << "Total parameter tensors: " << params.size() << std::endl;
    
    size_t total_params = 0;
    for (const auto& p : params) {
        total_params += p->getData().numel();
    }
    std::cout << "Total parameter count: " << total_params << std::endl;

    size_t without_tying = total_params + static_cast<size_t>(d_model) * vocab_size + vocab_size;
    
    std::cout << "\nWith weight tying: " << total_params << " parameters" << std::endl;
    std::cout << "Without tying would be: " << without_tying << " parameters" << std::endl;
    std::cout << "Saved: " << (without_tying - total_params) << " parameters" << std::endl;
    
    auto input = Variable::create(Tensor(1, 10), true);
    for (int i = 0; i < 10; i++) {
        input->getData().setValue(0, i, float(i % vocab_size));
    }
    
    std::cout << "\nRunning forward pass..." << std::endl;
    auto output = model.forward(input, false);
    
    std::cout << "Input shape: (1, " << input->getData().getCols() << ")" << std::endl;
    std::cout << "Output shape: (" << output->getData().getRows() << ", " << output->getData().getCols() << ")" << std::endl;
    std::cout << "Expected output cols: " << vocab_size << std::endl;

    CHECK(output->getData().getCols() == static_cast<size_t>(vocab_size));

    // Tied: the embedding table is the only vocab x d_model parameter.
    const auto table = model.getTokenEmbedding().getEmbeddingTable();
    int vocab_by_d = 0;
    int table_refs = 0;
    for (const auto& p : params) {
        if (p->getData().numel() == static_cast<size_t>(vocab_size) * d_model) vocab_by_d++;
        if (p == table) table_refs++;
    }
    CHECK(table_refs == 1);
    CHECK(vocab_by_d == 1);

    // Behavioral check: only tokens 0..9 appear in the input, so an untied
    // input embedding would give rows 10.. zero gradient. Through the tied
    // output projection every row receives gradient.
    Tensor ids(10, 1), targets(10, 1);
    for (int i = 0; i < 10; i++) {
        ids.setValue(i, 0, static_cast<float>(i));
        targets.setValue(i, 0, static_cast<float>((i + 1) % 10));
    }
    for (const auto& p : params) p->zeroGrad();
    auto loss = model.forward(Variable::create(ids, false), false)
                    ->log_softmax()->nll_loss(Variable::create(targets, false));
    loss->backward();
    const Tensor& grad = table->getGrad();
    bool unused_rows_have_grad = true;
    for (int row = 10; row < vocab_size; row++) {
        float row_abs = 0.0f;
        for (int c = 0; c < d_model; c++) row_abs += std::abs(grad.getValue(row, c));
        if (!(row_abs > 0.0f)) unused_rows_have_grad = false;
    }
    loss->release_graph();
    CHECK(unused_rows_have_grad);

    return test_util::exit_code();
}
