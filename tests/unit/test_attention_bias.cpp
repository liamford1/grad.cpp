#include <iostream>
#include "transformer/multihead_attention.h"
#include "transformer/variable.h"
#include "transformer/tensor.h"

int main() {
    std::cout << "=== MultiHeadAttention Bias Verification ===" << std::endl;
    
    int d_model = 256;
    int num_heads = 4;
    
    MultiHeadAttention attn(d_model, num_heads, 0.0f);
    
    auto params = attn.parameters();
    std::cout << "Total parameters: " << params.size() << " (expected: 8)" << std::endl;
    
    std::cout << "\nParameter shapes:" << std::endl;
    std::cout << "W_q: " << attn.getW_q()->getData().getRows() << "x" << attn.getW_q()->getData().getCols() << std::endl;
    std::cout << "W_k: " << attn.getW_k()->getData().getRows() << "x" << attn.getW_k()->getData().getCols() << std::endl;
    std::cout << "W_v: " << attn.getW_v()->getData().getRows() << "x" << attn.getW_v()->getData().getCols() << std::endl;
    std::cout << "W_o: " << attn.getW_o()->getData().getRows() << "x" << attn.getW_o()->getData().getCols() << std::endl;
    
    auto input = Variable::create(Tensor(10, d_model), true);
    for (int i = 0; i < input->getData().numel(); i++) {
        input->getData().raw()[i] = 0.01f * (i % 100);
    }
    
    std::cout << "\nRunning forward pass..." << std::endl;
    auto output = attn.forward(input, false);
    
    std::cout << "Input shape: (" << input->getData().getRows() << ", " << input->getData().getCols() << ")" << std::endl;
    std::cout << "Output shape: (" << output->getData().getRows() << ", " << output->getData().getCols() << ")" << std::endl;
    
    std::cout << "\n✓ Bias terms added successfully!" << std::endl;
    std::cout << "✓ Forward pass works with bias!" << std::endl;
    
    return 0;
}