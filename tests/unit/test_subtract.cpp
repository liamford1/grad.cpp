#include "transformer/tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>

void test_basic_subtract() {
    std::cout << "Test 1: Basic Subtraction\n";
    
    Tensor a(1, 2);
    a.setValue(0, 0, 5.0f);
    a.setValue(0, 1, 3.0f);
    
    Tensor b(1, 2);
    b.setValue(0, 0, 2.0f);
    b.setValue(0, 1, 1.0f);
    
    Tensor c = a.subtract(b);
    
    // Expected: c = a - b = [5-2, 3-1] = [3.0, 2.0]
    float expected_0 = 3.0f;
    float expected_1 = 2.0f;
    
    float actual_0 = c.getValue(0, 0);
    float actual_1 = c.getValue(0, 1);
    
    std::cout << "  a = [" << a.getValue(0, 0) << ", " << a.getValue(0, 1) << "]\n";
    std::cout << "  b = [" << b.getValue(0, 0) << ", " << b.getValue(0, 1) << "]\n";
    std::cout << "  Expected: c = a - b = [" << expected_0 << ", " << expected_1 << "]\n";
    std::cout << "  Actual:   c = [" << actual_0 << ", " << actual_1 << "]\n";
    
    assert(std::abs(actual_0 - expected_0) < 1e-6f);
    assert(std::abs(actual_1 - expected_1) < 1e-6f);
    
    std::cout << "  ✓ PASSED\n\n";
}

void test_negative_numbers() {
    std::cout << "Test 2: Subtraction with Negative Numbers\n";
    
    Tensor x(1, 3);
    x.setValue(0, 0, -5.0f);
    x.setValue(0, 1, 10.0f);
    x.setValue(0, 2, 0.0f);
    
    Tensor y(1, 3);
    y.setValue(0, 0, 3.0f);
    y.setValue(0, 1, -2.0f);
    y.setValue(0, 2, -1.0f);
    
    Tensor z = x.subtract(y);
    // Expected: z = x - y = [-5-3, 10-(-2), 0-(-1)] = [-8.0, 12.0, 1.0]
    
    std::cout << "  x = [" << x.getValue(0, 0) << ", " << x.getValue(0, 1) << ", " << x.getValue(0, 2) << "]\n";
    std::cout << "  y = [" << y.getValue(0, 0) << ", " << y.getValue(0, 1) << ", " << y.getValue(0, 2) << "]\n";
    std::cout << "  Expected: z = [-8.0, 12.0, 1.0]\n";
    std::cout << "  Actual:   z = [" << z.getValue(0, 0) << ", " << z.getValue(0, 1) << ", " << z.getValue(0, 2) << "]\n";
    
    assert(std::abs(z.getValue(0, 0) - (-8.0f)) < 1e-6f);
    assert(std::abs(z.getValue(0, 1) - 12.0f) < 1e-6f);
    assert(std::abs(z.getValue(0, 2) - 1.0f) < 1e-6f);
    
    std::cout << "  ✓ PASSED\n\n";
}

void test_3d_subtract() {
    std::cout << "Test 3: 3D Tensor Subtraction\n";
    
    Tensor a(2, 2, 2);  // batch=2, rows=2, cols=2
    // Batch 0
    a.setValue(0, 0, 0, 10.0f);
    a.setValue(0, 0, 1, 20.0f);
    a.setValue(0, 1, 0, 30.0f);
    a.setValue(0, 1, 1, 40.0f);
    // Batch 1
    a.setValue(1, 0, 0, 50.0f);
    a.setValue(1, 0, 1, 60.0f);
    a.setValue(1, 1, 0, 70.0f);
    a.setValue(1, 1, 1, 80.0f);
    
    Tensor b(2, 2, 2);
    // Batch 0
    b.setValue(0, 0, 0, 1.0f);
    b.setValue(0, 0, 1, 2.0f);
    b.setValue(0, 1, 0, 3.0f);
    b.setValue(0, 1, 1, 4.0f);
    // Batch 1
    b.setValue(1, 0, 0, 5.0f);
    b.setValue(1, 0, 1, 6.0f);
    b.setValue(1, 1, 0, 7.0f);
    b.setValue(1, 1, 1, 8.0f);
    
    Tensor c = a.subtract(b);
    
    // Expected: c = a - b
    // Batch 0: [10-1, 20-2, 30-3, 40-4] = [9, 18, 27, 36]
    // Batch 1: [50-5, 60-6, 70-7, 80-8] = [45, 54, 63, 72]
    
    std::cout << "  Checking batch 0...\n";
    assert(std::abs(c.getValue(0, 0, 0) - 9.0f) < 1e-6f);
    assert(std::abs(c.getValue(0, 0, 1) - 18.0f) < 1e-6f);
    assert(std::abs(c.getValue(0, 1, 0) - 27.0f) < 1e-6f);
    assert(std::abs(c.getValue(0, 1, 1) - 36.0f) < 1e-6f);
    
    std::cout << "  Checking batch 1...\n";
    assert(std::abs(c.getValue(1, 0, 0) - 45.0f) < 1e-6f);
    assert(std::abs(c.getValue(1, 0, 1) - 54.0f) < 1e-6f);
    assert(std::abs(c.getValue(1, 1, 0) - 63.0f) < 1e-6f);
    assert(std::abs(c.getValue(1, 1, 1) - 72.0f) < 1e-6f);
    
    std::cout << "  ✓ PASSED\n\n";
}

void test_layernorm_scenario() {
    std::cout << "Test 4: LayerNorm Scenario (x - mean)\n";
    
    // Simulating LayerNorm: normalized = (x - mean) / std
    Tensor x(1, 4);
    x.setValue(0, 0, 1.0f);
    x.setValue(0, 1, 2.0f);
    x.setValue(0, 2, 3.0f);
    x.setValue(0, 3, 4.0f);
    
    // mean = (1+2+3+4)/4 = 2.5
    Tensor mean(1, 4);
    mean.fill(2.5f);
    
    Tensor centered = x.subtract(mean);
    // Expected: [1-2.5, 2-2.5, 3-2.5, 4-2.5] = [-1.5, -0.5, 0.5, 1.5]
    
    std::cout << "  x = [1.0, 2.0, 3.0, 4.0]\n";
    std::cout << "  mean = 2.5\n";
    std::cout << "  Expected: x - mean = [-1.5, -0.5, 0.5, 1.5]\n";
    std::cout << "  Actual: [" << centered.getValue(0, 0) << ", " 
              << centered.getValue(0, 1) << ", "
              << centered.getValue(0, 2) << ", "
              << centered.getValue(0, 3) << "]\n";
    
    assert(std::abs(centered.getValue(0, 0) - (-1.5f)) < 1e-6f);
    assert(std::abs(centered.getValue(0, 1) - (-0.5f)) < 1e-6f);
    assert(std::abs(centered.getValue(0, 2) - 0.5f) < 1e-6f);
    assert(std::abs(centered.getValue(0, 3) - 1.5f) < 1e-6f);
    
    std::cout << "  ✓ PASSED\n\n";
}

void test_gradient_update_scenario() {
    std::cout << "Test 5: Gradient Update Scenario (param - lr * grad)\n";
    
    // Simulating: new_param = param - learning_rate * gradient
    Tensor param(1, 3);
    param.setValue(0, 0, 1.0f);
    param.setValue(0, 1, 2.0f);
    param.setValue(0, 2, 3.0f);
    
    Tensor lr_grad(1, 3);
    lr_grad.setValue(0, 0, 0.1f);  // lr * grad
    lr_grad.setValue(0, 1, 0.2f);
    lr_grad.setValue(0, 2, 0.3f);
    
    Tensor new_param = param.subtract(lr_grad);
    // Expected: [1-0.1, 2-0.2, 3-0.3] = [0.9, 1.8, 2.7]
    
    std::cout << "  param = [1.0, 2.0, 3.0]\n";
    std::cout << "  lr * grad = [0.1, 0.2, 0.3]\n";
    std::cout << "  Expected: new_param = [0.9, 1.8, 2.7]\n";
    std::cout << "  Actual: [" << new_param.getValue(0, 0) << ", " 
              << new_param.getValue(0, 1) << ", "
              << new_param.getValue(0, 2) << "]\n";
    
    assert(std::abs(new_param.getValue(0, 0) - 0.9f) < 1e-6f);
    assert(std::abs(new_param.getValue(0, 1) - 1.8f) < 1e-6f);
    assert(std::abs(new_param.getValue(0, 2) - 2.7f) < 1e-6f);
    
    std::cout << "  ✓ PASSED\n\n";
}

void test_commutativity() {
    std::cout << "Test 6: Non-Commutativity (a-b ≠ b-a)\n";
    
    Tensor a(1, 2);
    a.setValue(0, 0, 5.0f);
    a.setValue(0, 1, 3.0f);
    
    Tensor b(1, 2);
    b.setValue(0, 0, 2.0f);
    b.setValue(0, 1, 1.0f);
    
    Tensor a_minus_b = a.subtract(b);  // Should be [3, 2]
    Tensor b_minus_a = b.subtract(a);  // Should be [-3, -2]
    
    std::cout << "  a - b = [" << a_minus_b.getValue(0, 0) << ", " << a_minus_b.getValue(0, 1) << "]\n";
    std::cout << "  b - a = [" << b_minus_a.getValue(0, 0) << ", " << b_minus_a.getValue(0, 1) << "]\n";
    std::cout << "  Expected: a - b = [3, 2] and b - a = [-3, -2]\n";
    
    assert(std::abs(a_minus_b.getValue(0, 0) - 3.0f) < 1e-6f);
    assert(std::abs(a_minus_b.getValue(0, 1) - 2.0f) < 1e-6f);
    assert(std::abs(b_minus_a.getValue(0, 0) - (-3.0f)) < 1e-6f);
    assert(std::abs(b_minus_a.getValue(0, 1) - (-2.0f)) < 1e-6f);
    
    std::cout << "  ✓ PASSED\n\n";
}

int main() {
    std::cout << "=== COMPREHENSIVE SUBTRACT() TEST SUITE ===\n\n";
    
    try {
        test_basic_subtract();
        test_negative_numbers();
        test_3d_subtract();
        test_layernorm_scenario();
        test_gradient_update_scenario();
        test_commutativity();
        
        std::cout << "===========================================\n";
        std::cout << "ALL TESTS PASSED! ✓\n";
        std::cout << "The subtract() method is working correctly.\n";
        std::cout << "===========================================\n";
        
    } catch (const std::exception& e) {
        std::cout << "\n===========================================\n";
        std::cout << "TEST FAILED! ✗\n";
        std::cout << "Error: " << e.what() << "\n";
        std::cout << "===========================================\n";
        return 1;
    }
    
    return 0;
}