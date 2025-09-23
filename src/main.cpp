#include "tensor.h"
#include "attention.h"
#include "multihead_attention.h"
#include <iostream>

int main() {
    // =================================================================
    // TENSOR TESTS
    // =================================================================
    
    std::cout << "=== Testing Matrix Multiplication ===" << std::endl;
    Tensor matmul_a(2, 3); 
    Tensor matmul_b(3, 2);
    
    matmul_a.setValue(0, 0, 1); matmul_a.setValue(0, 1, 2); matmul_a.setValue(0, 2, 3);
    matmul_a.setValue(1, 0, 4); matmul_a.setValue(1, 1, 5); matmul_a.setValue(1, 2, 6);
    
    matmul_b.setValue(0, 0, 7); matmul_b.setValue(0, 1, 8);
    matmul_b.setValue(1, 0, 9); matmul_b.setValue(1, 1, 10);
    matmul_b.setValue(2, 0, 11); matmul_b.setValue(2, 1, 12);
    
    std::cout << "Matrix A:" << std::endl;
    matmul_a.display();
    
    std::cout << "\nMatrix B:" << std::endl;
    matmul_b.display();
    
    std::cout << "\nA * B:" << std::endl;
    Tensor matmul_c = matmul_a.matmul(matmul_b);
    matmul_c.display();

    // -----------------------------------------------------------------
    std::cout << "\n=== Testing Addition ===" << std::endl;
    Tensor add_x(2, 2);
    Tensor add_y(2, 2);

    add_x.setValue(0, 0, 1); add_x.setValue(0, 1, 2);
    add_x.setValue(1, 0, 3); add_x.setValue(1, 1, 4);

    add_y.setValue(0, 0, 5); add_y.setValue(0, 1, 6);
    add_y.setValue(1, 0, 7); add_y.setValue(1, 1, 8);

    std::cout << "Matrix X:" << std::endl;
    add_x.display();

    std::cout << "\nMatrix Y:" << std::endl;
    add_y.display();

    std::cout << "\nX + Y:" << std::endl;
    Tensor add_sum = add_x.add(add_y);
    add_sum.display();

    // -----------------------------------------------------------------
    std::cout << "\n=== Testing Transpose ===" << std::endl;
    Tensor transpose_t(2, 3);

    transpose_t.setValue(0, 0, 1); transpose_t.setValue(0, 1, 2); transpose_t.setValue(0, 2, 3);
    transpose_t.setValue(1, 0, 4); transpose_t.setValue(1, 1, 5); transpose_t.setValue(1, 2, 6);

    std::cout << "Original (2x3):" << std::endl;
    transpose_t.display();

    Tensor transpose_result = transpose_t.transpose();
    std::cout << "\nTransposed (3x2):" << std::endl;
    transpose_result.display();

    // -----------------------------------------------------------------
    std::cout << "\n=== Testing ReLU ===" << std::endl;
    Tensor relu_r(2, 3);

    relu_r.setValue(0, 0, 2.5f);  relu_r.setValue(0, 1, -1.2f); relu_r.setValue(0, 2, 0.0f);
    relu_r.setValue(1, 0, -3.7f); relu_r.setValue(1, 1, 5.1f);  relu_r.setValue(1, 2, -0.5f);

    std::cout << "Before ReLU:" << std::endl;
    relu_r.display();

    Tensor relu_result = relu_r.relu();
    std::cout << "\nAfter ReLU:" << std::endl;
    relu_result.display();

    // -----------------------------------------------------------------
    std::cout << "\n=== Testing Softmax ===" << std::endl;
    Tensor softmax_s(2, 3);

    softmax_s.setValue(0, 0, 0.0f); softmax_s.setValue(0, 1, 1.0f); softmax_s.setValue(0, 2, 0.0f);
    softmax_s.setValue(1, 0, 5.0f); softmax_s.setValue(1, 1, 1.0f); softmax_s.setValue(1, 2, 1.0f);

    std::cout << "Before softmax:" << std::endl;
    softmax_s.display();

    Tensor softmax_result = softmax_s.softmax();
    std::cout << "\nAfter softmax:" << std::endl;
    softmax_result.display();

    // -----------------------------------------------------------------
    std::cout << "\n=== Testing Fill ===" << std::endl;
    Tensor fill_w(2, 2);
    fill_w.fill(7.5f);
    std::cout << "Filled with 7.5:" << std::endl;
    fill_w.display();
    
    // -----------------------------------------------------------------
    std::cout << "\n=== Testing Scale ===" << std::endl;
    Tensor scale_tt(2, 2);
    scale_tt.setValue(0, 0, 2); scale_tt.setValue(0, 1, 4);
    scale_tt.setValue(1, 0, 6); scale_tt.setValue(1, 1, 8);

    std::cout << "Original:" << std::endl;
    scale_tt.display();

    Tensor scale_result = scale_tt.scale(0.5f);
    std::cout << "Scaled by 0.5:" << std::endl;
    scale_result.display();

    // -----------------------------------------------------------------
    std::cout << "\n=== Testing Copy Problem (Important!) ===" << std::endl;
    Tensor copy_original(2, 2);
    copy_original.fill(5.0);
    std::cout << "Original tensor filled with 5.0:" << std::endl;
    copy_original.display();

    Tensor copy_test = copy_original;  // This creates a copy
    copy_test.fill(10.0);
    std::cout << "\nAfter copying to 'copy_test' and filling 'copy_test' with 10.0:" << std::endl;
    std::cout << "Tensor 'copy_original' now shows:" << std::endl;
    copy_original.display();
    std::cout << "Tensor 'copy_test' shows:" << std::endl;
    copy_test.display();

    std::cout << "\nDid changing 'copy_test' also change 'copy_original'? " << 
        (copy_original.getValue(0,0) == 10.0 ? "YES (This is the bug!)" : "NO (Good!)") << std::endl;

    // =================================================================
    // ATTENTION TESTS
    // =================================================================
    
    std::cout << "\n\n=== Testing Single-Head Attention ===" << std::endl;
    
    // Create attention layer with small dimensions
    int attention_d_model = 4;  // Input/output dimension (renamed to avoid conflict)
    int attention_d_k = 3;      // Key/query dimension  
    int attention_d_v = 2;      // Value dimension
    
    Attention attention(attention_d_model, attention_d_k, attention_d_v);
    
    // Create simple input: 2 sequence positions, each of dimension d_model
    Tensor attention_input(2, attention_d_model);
    
    // Fill input with simple test values
    attention_input.setValue(0, 0, 1.0f); attention_input.setValue(0, 1, 0.0f); attention_input.setValue(0, 2, 1.0f); attention_input.setValue(0, 3, 0.0f);
    attention_input.setValue(1, 0, 0.0f); attention_input.setValue(1, 1, 1.0f); attention_input.setValue(1, 2, 0.0f); attention_input.setValue(1, 3, 1.0f);
    
    std::cout << "Input (2x" << attention_d_model << "):" << std::endl;
    attention_input.display();
    
    // Run attention
    Tensor attention_output = attention.forward(attention_input);
    
    std::cout << "\nAttention Output (should be 2x" << attention_d_model << "):" << std::endl;
    attention_output.display();
    
    // Verify output dimensions
    std::cout << "\nDimension check:" << std::endl;
    std::cout << "Input shape: 2x" << attention_d_model << std::endl;
    std::cout << "Output shape: " << attention_output.getRows() << "x" << attention_output.getCols() << std::endl;
    std::cout << "Shapes match: " << (attention_output.getRows() == 2 && attention_output.getCols() == attention_d_model ? "YES" : "NO") << std::endl;

    // -----------------------------------------------------------------
    std::cout << "\n=== Testing Reshape ===" << std::endl;
    Tensor reshape_original(2, 3);

    // Fill with sequential values to track ordering
    reshape_original.setValue(0, 0, 1); reshape_original.setValue(0, 1, 2); reshape_original.setValue(0, 2, 3);
    reshape_original.setValue(1, 0, 4); reshape_original.setValue(1, 1, 5); reshape_original.setValue(1, 2, 6);

    std::cout << "Original (2x3):" << std::endl;
    reshape_original.display();

    Tensor reshape_reshaped = reshape_original.reshape(3, 2);
    std::cout << "\nReshaped to (3x2):" << std::endl;
    reshape_reshaped.display();

    Tensor reshape_back = reshape_reshaped.reshape(1, 6);
    std::cout << "\nReshaped to (1x6):" << std::endl;
    reshape_back.display();

    // Test error case
    std::cout << "\nTesting invalid reshape:" << std::endl;
    try {
        Tensor reshape_invalid = reshape_original.reshape(2, 4); // Should fail: 2*3 != 2*4
        std::cout << "ERROR: Should have thrown exception!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Correctly caught exception: " << e.what() << std::endl;
    }

    // -----------------------------------------------------------------
    std::cout << "\n=== Testing Slice ===" << std::endl;
    Tensor slice_matrix(4, 5);

    // Fill with distinctive pattern (row*10 + col)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 5; j++) {
            slice_matrix.setValue(i, j, i * 10 + j);
        }
    }

    std::cout << "Original (4x5):" << std::endl;
    slice_matrix.display();

    // Test basic slice - get middle 2x3 portion
    Tensor slice1 = slice_matrix.slice(1, 2, 1, 3);
    std::cout << "\nSlice(1,2,1,3) - 2x3 from middle:" << std::endl;
    slice1.display();

    // Test corner slice
    Tensor slice2 = slice_matrix.slice(0, 2, 0, 2);
    std::cout << "\nSlice(0,2,0,2) - top-left 2x2:" << std::endl;
    slice2.display();

    // Test single row
    Tensor slice3 = slice_matrix.slice(2, 1, 0, 5);
    std::cout << "\nSlice(2,1,0,5) - entire row 2:" << std::endl;
    slice3.display();

    // Test error handling
    std::cout << "\nTesting slice out of bounds:" << std::endl;
    try {
        Tensor slice_invalid = slice_matrix.slice(3, 3, 0, 2); // Would go to row 6, but matrix only has 4 rows
        std::cout << "ERROR: Should have thrown exception!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Correctly caught exception: " << e.what() << std::endl;
    }

    // -----------------------------------------------------------------
    std::cout << "\n=== Testing Concatenate ===" << std::endl;
    Tensor concat_mat1(2, 3);
    Tensor concat_mat2(2, 3);

    concat_mat1.setValue(0, 0, 1); concat_mat1.setValue(0, 1, 2); concat_mat1.setValue(0, 2, 3);
    concat_mat1.setValue(1, 0, 4); concat_mat1.setValue(1, 1, 5); concat_mat1.setValue(1, 2, 6);

    concat_mat2.setValue(0, 0, 7); concat_mat2.setValue(0, 1, 8); concat_mat2.setValue(0, 2, 9);
    concat_mat2.setValue(1, 0, 10); concat_mat2.setValue(1, 1, 11); concat_mat2.setValue(1, 2, 12);

    std::cout << "Matrix 1 (2x3):" << std::endl;
    concat_mat1.display();

    std::cout << "\nMatrix 2 (2x3):" << std::endl;
    concat_mat2.display();

    Tensor concat_vertical = concat_mat1.concatenate(concat_mat2, 0);
    std::cout << "\nConcatenate axis=0 (vertical stack, result 4x3):" << std::endl;
    concat_vertical.display();

    Tensor concat_horizontal = concat_mat1.concatenate(concat_mat2, 1);
    std::cout << "\nConcatenate axis=1 (horizontal stack, result 2x6):" << std::endl;
    concat_horizontal.display();

    std::cout << "\nTesting concatenate with different sized matrices:" << std::endl;
    Tensor concat_tall(3, 2);
    Tensor concat_wide(1, 2);

    concat_tall.setValue(0, 0, 1); concat_tall.setValue(0, 1, 2);
    concat_tall.setValue(1, 0, 3); concat_tall.setValue(1, 1, 4);
    concat_tall.setValue(2, 0, 5); concat_tall.setValue(2, 1, 6);

    concat_wide.setValue(0, 0, 7); concat_wide.setValue(0, 1, 8);

    std::cout << "Tall matrix (3x2):" << std::endl;
    concat_tall.display();

    std::cout << "\nWide matrix (1x2):" << std::endl;
    concat_wide.display();

    Tensor concat_mixed = concat_tall.concatenate(concat_wide, 0);
    std::cout << "\nConcatenate tall+wide axis=0 (result 4x2):" << std::endl;
    concat_mixed.display();

    std::cout << "\nTesting concatenate error cases:" << std::endl;
    try {
        Tensor concat_incompatible1 = concat_mat1.concatenate(concat_tall, 0);
        std::cout << "ERROR: Should have thrown exception!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Correctly caught axis=0 dimension mismatch: " << e.what() << std::endl;
    }

    try {
        Tensor concat_incompatible2 = concat_mat1.concatenate(concat_tall, 1);
        std::cout << "ERROR: Should have thrown exception!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Correctly caught axis=1 dimension mismatch: " << e.what() << std::endl;
    }

    try {
        Tensor concat_invalid_axis = concat_mat1.concatenate(concat_mat2, 2);
        std::cout << "ERROR: Should have thrown exception!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Correctly caught invalid axis: " << e.what() << std::endl;
    }

    // =================================================================
    // MULTI-HEAD ATTENTION TESTS
    // =================================================================
    
    std::cout << "\n\n=== Testing Multi-Head Attention ===" << std::endl;
    
    // Test parameters (renamed to avoid conflicts)
    int mha_seq_len = 4;      // Short sequence for easy verification
    int mha_d_model = 8;      // Small model size
    int mha_num_heads = 2;    // 2 heads, so each head gets d_k = 4
    
    // Create test input with varied values (not all identical)
    Tensor mha_input(mha_seq_len, mha_d_model);
    
    // Row 0: alternating pattern
    mha_input.setValue(0, 0, 1.0); mha_input.setValue(0, 1, 0.0); mha_input.setValue(0, 2, 1.0); mha_input.setValue(0, 3, 0.0);
    mha_input.setValue(0, 4, 0.5); mha_input.setValue(0, 5, 0.5); mha_input.setValue(0, 6, 0.5); mha_input.setValue(0, 7, 0.5);
    
    // Row 1: reverse pattern
    mha_input.setValue(1, 0, 0.0); mha_input.setValue(1, 1, 1.0); mha_input.setValue(1, 2, 0.0); mha_input.setValue(1, 3, 1.0);
    mha_input.setValue(1, 4, 0.2); mha_input.setValue(1, 5, 0.8); mha_input.setValue(1, 6, 0.3); mha_input.setValue(1, 7, 0.7);
    
    // Row 2: moderate values
    mha_input.setValue(2, 0, 0.3); mha_input.setValue(2, 1, 0.3); mha_input.setValue(2, 2, 0.3); mha_input.setValue(2, 3, 0.3);
    mha_input.setValue(2, 4, 0.6); mha_input.setValue(2, 5, 0.6); mha_input.setValue(2, 6, 0.6); mha_input.setValue(2, 7, 0.6);
    
    // Row 3: higher values
    mha_input.setValue(3, 0, 0.8); mha_input.setValue(3, 1, 0.8); mha_input.setValue(3, 2, 0.8); mha_input.setValue(3, 3, 0.8);
    mha_input.setValue(3, 4, 0.9); mha_input.setValue(3, 5, 0.9); mha_input.setValue(3, 6, 0.9); mha_input.setValue(3, 7, 0.9);
    
    std::cout << "Input shape: [" << mha_input.getRows() << ", " << mha_input.getCols() << "]" << std::endl;
    std::cout << "Input:" << std::endl;
    mha_input.display();
    
    // Create multi-head attention
    MultiHeadAttention mha(mha_d_model, mha_num_heads);
    
    // Forward pass
    Tensor mha_output = mha.forward(mha_input);
    
    std::cout << "\nOutput shape: [" << mha_output.getRows() << ", " << mha_output.getCols() << "]" << std::endl;
    std::cout << "Output:" << std::endl;
    mha_output.display();
    
    // Verify dimensions
    if (mha_output.getRows() == mha_seq_len && mha_output.getCols() == mha_d_model) {
        std::cout << "\n✅ Output dimensions are correct!" << std::endl;
    } else {
        std::cout << "\n❌ Output dimensions are wrong!" << std::endl;
    }

    std::cout << "\n=== Testing Concatenation Bug ===" << std::endl;
    Tensor test_a(4, 4);
    Tensor test_b(4, 4);
    test_a.fill(0.5);
    test_b.fill(-0.5);

    std::cout << "Before concatenation:" << std::endl;
    std::cout << "test_a sample: " << test_a.getValue(0,0) << std::endl;
    std::cout << "test_b sample: " << test_b.getValue(0,0) << std::endl;

    Tensor concatenated = test_a.concatenate(test_b, 1);
    std::cout << "After concatenation sample: " << concatenated.getValue(0,0) << std::endl;
    std::cout << "Concatenated shape: [" << concatenated.getRows() << ", " << concatenated.getCols() << "]" << std::endl;
    concatenated.display(); // Show the whole matrix

    return 0;
}