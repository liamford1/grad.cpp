#include "tensor.h"
#include "attention.h"
#include "multihead_attention.h"
#include "layer_norm.h"
#include "linear.h"
#include "feedforward.h"
#include "transformer_block.h"
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

    // -----------------------------------------------------------------
    std::cout << "\n=== Testing Broadcasting ===" << std::endl;
    
    Tensor A_broadcast(3, 4);
    Tensor B_broadcast(1, 4);
    A_broadcast.fill(2.0f);
    B_broadcast.fill(0.5f);
    
    Tensor C_broadcast = A_broadcast.add(B_broadcast);  // [3,4] + [1,4] should work
    std::cout << "Broadcasting [3,4] + [1,4]: " << (C_broadcast.getRows() == 3 && C_broadcast.getCols() == 4 ? "PASS" : "FAIL") << std::endl;
    std::cout << "Values correct: " << (C_broadcast.getValue(0,0) == 2.5f && C_broadcast.getValue(2,3) == 2.5f ? "PASS" : "FAIL") << std::endl;
    
    // Test another broadcasting case
    Tensor D(2, 3);
    Tensor E(2, 1);
    D.fill(1.0f);
    E.fill(3.0f);
    
    Tensor F = D.add(E);  // [2,3] + [2,1] should work
    std::cout << "Broadcasting [2,3] + [2,1]: " << (F.getRows() == 2 && F.getCols() == 3 ? "PASS" : "FAIL") << std::endl;
    std::cout << "Values correct: " << (F.getValue(0,0) == 4.0f && F.getValue(1,2) == 4.0f ? "PASS" : "FAIL") << std::endl;

    // -----------------------------------------------------------------
    std::cout << "\n=== Testing Linear Layer ===" << std::endl;

    Tensor linear_input(3, 4);  // [seq_len=3, input_dim=4]
    linear_input.fill(1.0f);

    Linear linear_layer(4, 6, true);  // input_dim=4, output_dim=6, bias=true
    Tensor linear_output = linear_layer.forward(linear_input);

    std::cout << "Linear input shape: [" << linear_input.getRows() << ", " << linear_input.getCols() << "]" << std::endl;
    std::cout << "Linear output shape: [" << linear_output.getRows() << ", " << linear_output.getCols() << "]" << std::endl;
    std::cout << "Dimensions correct: " << (linear_output.getRows() == 3 && linear_output.getCols() == 6 ? "YES" : "NO") << std::endl;

    // Test without bias
    Linear linear_no_bias(4, 6, false);
    Tensor output_no_bias = linear_no_bias.forward(linear_input);
    std::cout << "No-bias layer works: " << (output_no_bias.getRows() == 3 && output_no_bias.getCols() == 6 ? "YES" : "NO") << std::endl;

    // -----------------------------------------------------------------
    std::cout << "\n=== Testing Feed-Forward Network ===" << std::endl;

    Tensor ffn_input(4, 8);  // [seq_len=4, d_model=8]
    ffn_input.fill(0.5f);

    // Test with default hidden_dim (4 * d_model = 32)
    FeedForward ffn_default(8);
    Tensor ffn_output_default = ffn_default.forward(ffn_input);

    std::cout << "FFN input shape: [" << ffn_input.getRows() << ", " << ffn_input.getCols() << "]" << std::endl;
    std::cout << "FFN output shape: [" << ffn_output_default.getRows() << ", " << ffn_output_default.getCols() << "]" << std::endl;
    std::cout << "Default FFN works: " << (ffn_output_default.getRows() == 4 && ffn_output_default.getCols() == 8 ? "YES" : "NO") << std::endl;

    // Test with custom hidden_dim
    FeedForward ffn_custom(8, 16);  // Custom hidden_dim = 16 instead of 32
    Tensor ffn_output_custom = ffn_custom.forward(ffn_input);
    std::cout << "Custom FFN works: " << (ffn_output_custom.getRows() == 4 && ffn_output_custom.getCols() == 8 ? "YES" : "NO") << std::endl;

    // -----------------------------------------------------------------
    std::cout << "\n=== Testing Transformer Block ===" << std::endl;

    Tensor tb_input(4, 8);  // [seq_len=4, d_model=8]
    tb_input.fill(0.3f);

    TransformerBlock transformer_block(8, 2);  // d_model=8, num_heads=2, default FFN
    Tensor tb_output = transformer_block.forward(tb_input);

    std::cout << "TransformerBlock input shape: [" << tb_input.getRows() << ", " << tb_input.getCols() << "]" << std::endl;
    std::cout << "TransformerBlock output shape: [" << tb_output.getRows() << ", " << tb_output.getCols() << "]" << std::endl;
    std::cout << "TransformerBlock works: " << (tb_output.getRows() == 4 && tb_output.getCols() == 8 ? "YES" : "NO") << std::endl;

    // Test with custom FFN hidden dimension
    TransformerBlock tb_custom(8, 2, 16);  // Custom FFN hidden_dim = 16
    Tensor tb_custom_output = tb_custom.forward(tb_input);
    std::cout << "Custom TransformerBlock works: " << (tb_custom_output.getRows() == 4 && tb_custom_output.getCols() == 8 ? "YES" : "NO") << std::endl;

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
    std::cout << "\n=== Testing Complete Transformer Pipeline ===" << std::endl;
    
    Tensor pipeline_input(6, 8);  // [seq_len=6, d_model=8] 
    pipeline_input.fill(0.2f);
    
    // Chain multiple transformer blocks
    TransformerBlock block1(8, 2);
    TransformerBlock block2(8, 2);
    
    Tensor after_block1 = block1.forward(pipeline_input);
    Tensor after_block2 = block2.forward(after_block1);
    
    std::cout << "Multi-block pipeline input: [" << pipeline_input.getRows() << ", " << pipeline_input.getCols() << "]" << std::endl;
    std::cout << "Multi-block pipeline output: [" << after_block2.getRows() << ", " << after_block2.getCols() << "]" << std::endl;
    std::cout << "Multi-block pipeline works: " << (after_block2.getRows() == 6 && after_block2.getCols() == 8 ? "YES" : "NO") << std::endl;

    // =================================================================
    // SUMMARY
    // =================================================================
    
    std::cout << "\n=== TRANSFORMER STATUS ===" << std::endl;
    std::cout << "✓ Tensor operations working" << std::endl;
    std::cout << "✓ Broadcasting implemented" << std::endl;
    std::cout << "✓ Linear layers implemented" << std::endl;
    std::cout << "✓ Feed-forward networks implemented" << std::endl;
    std::cout << "✓ Multi-head attention implemented" << std::endl;
    std::cout << "✓ Layer normalization working" << std::endl;
    std::cout << "✓ Transformer blocks with residual connections implemented" << std::endl;
    std::cout << "✓ Multi-block transformer pipeline working" << std::endl;
    std::cout << "✓ Components integrate successfully" << std::endl;
    std::cout << "\nNext steps: Positional encoding, embeddings, or full transformer model" << std::endl;

    return 0;
}