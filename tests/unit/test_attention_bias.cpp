#include <cmath>
#include <iostream>
#include "grad/transformer/multihead_attention.h"
#include "grad/transformer/variable.h"
#include "grad/transformer/tensor.h"
#include "../test_util.h"

using namespace grad;

int main() {
    std::cout << "=== MultiHeadAttention Bias Verification ===" << std::endl;
    
    int d_model = 256;
    int num_heads = 4;
    
    MultiHeadAttention attn(d_model, num_heads, 0.0f);
    
    auto params = attn.parameters();
    std::cout << "Total parameters: " << params.size() << " (expected: 8)" << std::endl;
    CHECK(params.size() == 8);
    
    std::cout << "\nParameter shapes:" << std::endl;
    std::cout << "W_q: " << attn.getW_q()->getData().getRows() << "x" << attn.getW_q()->getData().getCols() << std::endl;
    std::cout << "W_k: " << attn.getW_k()->getData().getRows() << "x" << attn.getW_k()->getData().getCols() << std::endl;
    std::cout << "W_v: " << attn.getW_v()->getData().getRows() << "x" << attn.getW_v()->getData().getCols() << std::endl;
    std::cout << "W_o: " << attn.getW_o()->getData().getRows() << "x" << attn.getW_o()->getData().getCols() << std::endl;

    const size_t d = static_cast<size_t>(d_model);
    for (const auto& w : {attn.getW_q(), attn.getW_k(), attn.getW_v(), attn.getW_o()}) {
        CHECK(w->getData().getRows() == d && w->getData().getCols() == d);
    }
    for (const auto& b : {attn.getB_q(), attn.getB_k(), attn.getB_v(), attn.getB_o()}) {
        CHECK(b != nullptr && b->getData().numel() == d);
    }
    
    auto input = Variable::create(Tensor(10, d), true);
    for (size_t i = 0; i < input->getData().numel(); i++) {
        input->getData().raw()[i] = 0.01f * static_cast<float>(i % 100);
    }
    
    std::cout << "\nRunning forward pass..." << std::endl;
    auto output = attn.forward(input, false);
    
    std::cout << "Input shape: (" << input->getData().getRows() << ", " << input->getData().getCols() << ")" << std::endl;
    std::cout << "Output shape: (" << output->getData().getRows() << ", " << output->getData().getCols() << ")" << std::endl;
    
    CHECK(output->getData().getRows() == 10);
    CHECK(output->getData().getCols() == d);

    // The bias must reach the output: with zero input every projection is
    // bias-only, and with b_v = 0 the attended values are zero, so the
    // output is exactly b_o on every row.
    auto zeros = Variable::create(Tensor(10, d), false);
    for (size_t i = 0; i < d; i++) {
        attn.getB_v()->getData().raw()[i] = 0.0f;
        attn.getB_o()->getData().raw()[i] = 0.5f;
    }
    auto biased = attn.forward(zeros, false);
    bool bias_reaches_output = true;
    for (size_t i = 0; i < biased->getData().numel(); i++) {
        if (std::abs(biased->getData().raw()[i] - 0.5f) > 1e-5f) bias_reaches_output = false;
    }
    CHECK(bias_reaches_output);

    return test_util::exit_code();
}
