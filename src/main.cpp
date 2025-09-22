#include "tensor.h"
#include <iostream>

int main() {
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

    // Test addition
    std::cout << "\n=== Testing Addition ===" << std::endl;
    Tensor x(2, 2);
    Tensor y(2, 2);

    // Fill x
    x.setValue(0, 0, 1); x.setValue(0, 1, 2);
    x.setValue(1, 0, 3); x.setValue(1, 1, 4);

    // Fill y  
    y.setValue(0, 0, 5); y.setValue(0, 1, 6);
    y.setValue(1, 0, 7); y.setValue(1, 1, 8);

    std::cout << "Matrix X:" << std::endl;
    x.display();

    std::cout << "\nMatrix Y:" << std::endl;
    y.display();

    std::cout << "\nX + Y:" << std::endl;
    Tensor sum = x.add(y);
    sum.display();

    // Test transpose
    std::cout << "\n=== Testing Transpose ===" << std::endl;
    Tensor t(2, 3);  // 2x3 matrix

    // Fill with simple values
    t.setValue(0, 0, 1); t.setValue(0, 1, 2); t.setValue(0, 2, 3);
    t.setValue(1, 0, 4); t.setValue(1, 1, 5); t.setValue(1, 2, 6);

    std::cout << "Original (2x3):" << std::endl;
    t.display();

    Tensor transposed = t.transpose();
    std::cout << "\nTransposed (3x2):" << std::endl;
    transposed.display();

    // Test ReLU
    std::cout << "\n=== Testing ReLU ===" << std::endl;
    Tensor r(2, 3);

    // Fill with mix of positive and negative values
    r.setValue(0, 0, 2.5f);  r.setValue(0, 1, -1.2f); r.setValue(0, 2, 0.0f);
    r.setValue(1, 0, -3.7f); r.setValue(1, 1, 5.1f);  r.setValue(1, 2, -0.5f);

    std::cout << "Before ReLU:" << std::endl;
    r.display();

    Tensor relu_result = r.relu();
    std::cout << "\nAfter ReLU:" << std::endl;
    relu_result.display();

    // Test softmax
    std::cout << "\n=== Testing Softmax ===" << std::endl;
    Tensor s(2, 3);

    // Fill with some values
    s.setValue(0, 0, 0.0f); s.setValue(0, 1, 1.0f); s.setValue(0, 2, 0.0f);  // Should favor middle
    s.setValue(1, 0, 5.0f); s.setValue(1, 1, 1.0f); s.setValue(1, 2, 1.0f);  // Should favor first

    std::cout << "Before softmax:" << std::endl;
    s.display();

    Tensor soft_result = s.softmax();
    std::cout << "\nAfter softmax:" << std::endl;
    soft_result.display();

    Tensor w(2, 2);
    w.fill(7.5f);
    w.display();  // Should show all 7.5s
    
    // Test scale
    Tensor tt(2, 2);
    tt.setValue(0, 0, 2); tt.setValue(0, 1, 4);
    tt.setValue(1, 0, 6); tt.setValue(1, 1, 8);

    std::cout << "Original:" << std::endl;
    tt.display();

    Tensor scaled = tt.scale(0.5f);
    std::cout << "Scaled by 0.5:" << std::endl;
    scaled.display();
    return 0;
}

