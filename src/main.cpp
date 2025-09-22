#include "tensor.h"
#include "attention.h"
#include <iostream>

int main() {
    // =================================================================
    // TENSOR TESTS
    // =================================================================
    
    std::cout << "=== Testing Matrix Multiplication ===" << std::endl;
    Tensor a(2, 3); 
    Tensor b(3, 2);
    
    a.setValue(0, 0, 1); a.setValue(0, 1, 2); a.setValue(0, 2, 3);
    a.setValue(1, 0, 4); a.setValue(1, 1, 5); a.setValue(1, 2, 6);
    
    b.setValue(0, 0, 7); b.setValue(0, 1, 8);
    b.setValue(1, 0, 9); b.setValue(1, 1, 10);
    b.setValue(2, 0, 11); b.setValue(2, 1, 12);
    
    std::cout << "Matrix A:" << std::endl;
    a.display();
    
    std::cout << "\nMatrix B:" << std::endl;
    b.display();
    
    std::cout << "\nA * B:" << std::endl;
    Tensor c = a.matmul(b);
    c.display();

    // -----------------------------------------------------------------
    std::cout << "\n=== Testing Addition ===" << std::endl;
    Tensor x(2, 2);
    Tensor y(2, 2);

    x.setValue(0, 0, 1); x.setValue(0, 1, 2);
    x.setValue(1, 0, 3); x.setValue(1, 1, 4);

    y.setValue(0, 0, 5); y.setValue(0, 1, 6);
    y.setValue(1, 0, 7); y.setValue(1, 1, 8);

    std::cout << "Matrix X:" << std::endl;
    x.display();

    std::cout << "\nMatrix Y:" << std::endl;
    y.display();

    std::cout << "\nX + Y:" << std::endl;
    Tensor sum = x.add(y);
    sum.display();

    // -----------------------------------------------------------------
    std::cout << "\n=== Testing Transpose ===" << std::endl;
    Tensor t(2, 3);

    t.setValue(0, 0, 1); t.setValue(0, 1, 2); t.setValue(0, 2, 3);
    t.setValue(1, 0, 4); t.setValue(1, 1, 5); t.setValue(1, 2, 6);

    std::cout << "Original (2x3):" << std::endl;
    t.display();

    Tensor transposed = t.transpose();
    std::cout << "\nTransposed (3x2):" << std::endl;
    transposed.display();

    // -----------------------------------------------------------------
    std::cout << "\n=== Testing ReLU ===" << std::endl;
    Tensor r(2, 3);

    r.setValue(0, 0, 2.5f);  r.setValue(0, 1, -1.2f); r.setValue(0, 2, 0.0f);
    r.setValue(1, 0, -3.7f); r.setValue(1, 1, 5.1f);  r.setValue(1, 2, -0.5f);

    std::cout << "Before ReLU:" << std::endl;
    r.display();

    Tensor relu_result = r.relu();
    std::cout << "\nAfter ReLU:" << std::endl;
    relu_result.display();

    // -----------------------------------------------------------------
    std::cout << "\n=== Testing Softmax ===" << std::endl;
    Tensor s(2, 3);

    s.setValue(0, 0, 0.0f); s.setValue(0, 1, 1.0f); s.setValue(0, 2, 0.0f);
    s.setValue(1, 0, 5.0f); s.setValue(1, 1, 1.0f); s.setValue(1, 2, 1.0f);

    std::cout << "Before softmax:" << std::endl;
    s.display();

    Tensor soft_result = s.softmax();
    std::cout << "\nAfter softmax:" << std::endl;
    soft_result.display();

    // -----------------------------------------------------------------
    std::cout << "\n=== Testing Fill ===" << std::endl;
    Tensor w(2, 2);
    w.fill(7.5f);
    std::cout << "Filled with 7.5:" << std::endl;
    w.display();
    
    // -----------------------------------------------------------------
    std::cout << "\n=== Testing Scale ===" << std::endl;
    Tensor tt(2, 2);
    tt.setValue(0, 0, 2); tt.setValue(0, 1, 4);
    tt.setValue(1, 0, 6); tt.setValue(1, 1, 8);

    std::cout << "Original:" << std::endl;
    tt.display();

    Tensor scaled = tt.scale(0.5f);
    std::cout << "Scaled by 0.5:" << std::endl;
    scaled.display();

    // =================================================================
    // ATTENTION TESTS
    // =================================================================
    
    std::cout << "\n\n=== Testing Single-Head Attention ===" << std::endl;
    
    // Create attention layer with small dimensions
    int d_model = 4;  // Input/output dimension
    int d_k = 3;      // Key/query dimension  
    int d_v = 2;      // Value dimension
    
    Attention attention(d_model, d_k, d_v);
    
    // Create simple input: 2 sequence positions, each of dimension d_model
    Tensor input(2, d_model);
    
    // Fill input with simple test values
    input.setValue(0, 0, 1.0f); input.setValue(0, 1, 0.0f); input.setValue(0, 2, 1.0f); input.setValue(0, 3, 0.0f);
    input.setValue(1, 0, 0.0f); input.setValue(1, 1, 1.0f); input.setValue(1, 2, 0.0f); input.setValue(1, 3, 1.0f);
    
    std::cout << "Input (2x" << d_model << "):" << std::endl;
    input.display();
    
    // Run attention
    Tensor output = attention.forward(input);
    
    std::cout << "\nAttention Output (should be 2x" << d_model << "):" << std::endl;
    output.display();
    
    // Verify output dimensions
    std::cout << "\nDimension check:" << std::endl;
    std::cout << "Input shape: 2x" << d_model << std::endl;
    std::cout << "Output shape: " << output.getRows() << "x" << output.getCols() << std::endl;
    std::cout << "Shapes match: " << (output.getRows() == 2 && output.getCols() == d_model ? "YES" : "NO") << std::endl;

    // -----------------------------------------------------------------
    std::cout << "\n=== Testing Reshape ===" << std::endl;
    Tensor original(2, 3);

    // Fill with sequential values to track ordering
    original.setValue(0, 0, 1); original.setValue(0, 1, 2); original.setValue(0, 2, 3);
    original.setValue(1, 0, 4); original.setValue(1, 1, 5); original.setValue(1, 2, 6);

    std::cout << "Original (2x3):" << std::endl;
    original.display();

    Tensor reshaped = original.reshape(3, 2);
    std::cout << "\nReshaped to (3x2):" << std::endl;
    reshaped.display();

    Tensor back = reshaped.reshape(1, 6);
    std::cout << "\nReshaped to (1x6):" << std::endl;
    back.display();

    // Test error case
    std::cout << "\nTesting invalid reshape:" << std::endl;
    try {
        Tensor invalid = original.reshape(2, 4); // Should fail: 2*3 != 2*4
        std::cout << "ERROR: Should have thrown exception!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Correctly caught exception: " << e.what() << std::endl;
    }

    // -----------------------------------------------------------------
    std::cout << "\n=== Testing Slice ===" << std::endl;
    Tensor matrix(4, 5);

    // Fill with distinctive pattern (row*10 + col)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 5; j++) {
            matrix.setValue(i, j, i * 10 + j);
        }
    }

    std::cout << "Original (4x5):" << std::endl;
    matrix.display();

    // Test basic slice - get middle 2x3 portion
    Tensor slice1 = matrix.slice(1, 2, 1, 3);
    std::cout << "\nSlice(1,2,1,3) - 2x3 from middle:" << std::endl;
    slice1.display();

    // Test corner slice
    Tensor slice2 = matrix.slice(0, 2, 0, 2);
    std::cout << "\nSlice(0,2,0,2) - top-left 2x2:" << std::endl;
    slice2.display();

    // Test single row
    Tensor slice3 = matrix.slice(2, 1, 0, 5);
    std::cout << "\nSlice(2,1,0,5) - entire row 2:" << std::endl;
    slice3.display();

    // Test error handling
    std::cout << "\nTesting slice out of bounds:" << std::endl;
    try {
        Tensor invalid = matrix.slice(3, 3, 0, 2); // Would go to row 6, but matrix only has 4 rows
        std::cout << "ERROR: Should have thrown exception!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Correctly caught exception: " << e.what() << std::endl;
    }

    return 0;
}