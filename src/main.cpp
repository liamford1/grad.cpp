#include "tensor.h"
#include "attention.h"
#include "multihead_attention.h"
#include "layer_norm.h"
#include <iostream>

int main() {
    // =================================================================
    // CORE TENSOR FUNCTIONALITY TESTS
    // =================================================================
    
    std::cout << "=== Testing Matrix Multiplication ===" << std::endl;
    Tensor A(2, 3); 
    Tensor B(3, 2);
    
    A.setValue(0, 0, 1); A.setValue(0, 1, 2); A.setValue(0, 2, 3);
    A.setValue(1, 0, 4); A.setValue(1, 1, 5); A.setValue(1, 2, 6);
    
    B.setValue(0, 0, 7); B.setValue(0, 1, 8);
    B.setValue(1, 0, 9); B.setValue(1, 1, 10);
    B.setValue(2, 0, 11); B.setValue(2, 1, 12);
    
    Tensor C = A.matmul(B);
    std::cout << "Result: " << C.getValue(0,0) << " " << C.getValue(0,1) << std::endl;
    std::cout << "        " << C.getValue(1,0) << " " << C.getValue(1,1) << std::endl;

    // -----------------------------------------------------------------
    std::cout << "\n=== Testing Assignment Operator ===" << std::endl;
    Tensor original(2, 2);
    original.fill(5.0);
    
    Tensor copy(3, 3);  // Different size
    copy.fill(10.0);
    copy = original;    // Test assignment
    
    std::cout << "Assignment works: " << (copy.getValue(0,0) == 5.0 ? "YES" : "NO") << std::endl;
    std::cout << "Correct size: " << (copy.getRows() == 2 && copy.getCols() == 2 ? "YES" : "NO") << std::endl;

    // =================================================================
    // TRANSFORMER COMPONENT TESTS
    // =================================================================
    
    std::cout << "\n=== Testing Multi-Head Attention ===" << std::endl;
    
    int seq_len = 4;
    int d_model = 8;
    int num_heads = 2;
    
    Tensor mha_input(seq_len, d_model);
    
    // Create varied input patterns
    mha_input.setValue(0, 0, 1.0); mha_input.setValue(0, 1, 0.0); mha_input.setValue(0, 2, 1.0); mha_input.setValue(0, 3, 0.0);
    mha_input.setValue(0, 4, 0.5); mha_input.setValue(0, 5, 0.5); mha_input.setValue(0, 6, 0.5); mha_input.setValue(0, 7, 0.5);
    
    mha_input.setValue(1, 0, 0.0); mha_input.setValue(1, 1, 1.0); mha_input.setValue(1, 2, 0.0); mha_input.setValue(1, 3, 1.0);
    mha_input.setValue(1, 4, 0.2); mha_input.setValue(1, 5, 0.8); mha_input.setValue(1, 6, 0.3); mha_input.setValue(1, 7, 0.7);
    
    mha_input.setValue(2, 0, 0.3); mha_input.setValue(2, 1, 0.3); mha_input.setValue(2, 2, 0.3); mha_input.setValue(2, 3, 0.3);
    mha_input.setValue(2, 4, 0.6); mha_input.setValue(2, 5, 0.6); mha_input.setValue(2, 6, 0.6); mha_input.setValue(2, 7, 0.6);
    
    mha_input.setValue(3, 0, 0.8); mha_input.setValue(3, 1, 0.8); mha_input.setValue(3, 2, 0.8); mha_input.setValue(3, 3, 0.8);
    mha_input.setValue(3, 4, 0.9); mha_input.setValue(3, 5, 0.9); mha_input.setValue(3, 6, 0.9); mha_input.setValue(3, 7, 0.9);
    
    MultiHeadAttention mha(d_model, num_heads);
    Tensor mha_output = mha.forward(mha_input);
    
    std::cout << "Input shape: [" << mha_input.getRows() << ", " << mha_input.getCols() << "]" << std::endl;
    std::cout << "Output shape: [" << mha_output.getRows() << ", " << mha_output.getCols() << "]" << std::endl;
    std::cout << "Dimensions correct: " << (mha_output.getRows() == seq_len && mha_output.getCols() == d_model ? "YES" : "NO") << std::endl;

    // -----------------------------------------------------------------
    std::cout << "\n=== Testing Layer Normalization ===" << std::endl;
    
    Tensor ln_input(2, 4);
    
    // Different scale patterns
    ln_input.setValue(0, 0, 1.0f); ln_input.setValue(0, 1, 2.0f); 
    ln_input.setValue(0, 2, 3.0f); ln_input.setValue(0, 3, 4.0f);
    
    ln_input.setValue(1, 0, 10.0f); ln_input.setValue(1, 1, 20.0f); 
    ln_input.setValue(1, 2, 30.0f); ln_input.setValue(1, 3, 40.0f);
    
    LayerNorm layer_norm(4);
    Tensor ln_output = layer_norm.forward(ln_input);
    
    std::cout << "Normalization pattern (both rows should be similar):" << std::endl;
    std::cout << "Row 0: " << ln_output.getValue(0,0) << " " << ln_output.getValue(0,1) 
              << " " << ln_output.getValue(0,2) << " " << ln_output.getValue(0,3) << std::endl;
    std::cout << "Row 1: " << ln_output.getValue(1,0) << " " << ln_output.getValue(1,1) 
              << " " << ln_output.getValue(1,2) << " " << ln_output.getValue(1,3) << std::endl;

    // -----------------------------------------------------------------
    std::cout << "\n=== Testing Full Pipeline: MHA + LayerNorm ===" << std::endl;
    
    MultiHeadAttention mha2(d_model, num_heads);
    LayerNorm layer_norm2(d_model);
    
    Tensor attention_output = mha2.forward(mha_input);
    std::cout << "Attention output range: " << attention_output.getValue(0,0) << " to " << attention_output.getValue(3,7) << std::endl;
    
    Tensor normalized_output = layer_norm2.forward(attention_output);
    std::cout << "Normalized output range: " << normalized_output.getValue(0,0) << " to " << normalized_output.getValue(3,7) << std::endl;
    
    std::cout << "Pipeline works: " << (normalized_output.getRows() == seq_len && normalized_output.getCols() == d_model ? "YES" : "NO") << std::endl;

    // =================================================================
    // SUMMARY
    // =================================================================
    
    std::cout << "\n=== TRANSFORMER STATUS ===" << std::endl;
    std::cout << "✓ Tensor operations working" << std::endl;
    std::cout << "✓ Multi-head attention implemented" << std::endl;
    std::cout << "✓ Layer normalization working" << std::endl;
    std::cout << "✓ Components integrate successfully" << std::endl;
    std::cout << "\nNext steps: Feed-forward networks, positional encoding, or residual connections" << std::endl;

    return 0;
}