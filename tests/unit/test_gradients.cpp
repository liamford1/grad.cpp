#include <iostream>
#include <cmath>
#include <iomanip>
#include "transformer/variable.h"
#include "transformer/tensor.h"

float numericalGradient(std::function<float()> forward_fn, float* param, float epsilon = 1e-5) {
    float original = *param;

    *param = original + epsilon;
    float loss_plus = forward_fn();

    *param = original - epsilon;
    float loss_minus = forward_fn();

    *param = original;

    return (loss_plus - loss_minus) / (2.0f * epsilon);
}

bool checkGradient(float analytical, float numerical, float tolerance = 1e-3) {
    float abs_error = std::abs(analytical - numerical);
    float denom = std::max(std::abs(analytical), std::abs(numerical));
    float rel_error;

    if (denom < 1e-8) {
        rel_error = abs_error;
    } else {
        rel_error = abs_error / denom;
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Analytical: " << std::setw(12) << analytical;
    std::cout << "  Numerical: " << std::setw(12) << numerical;
    std::cout << "  Rel Error: " << std::scientific << rel_error;
    
    bool passed = (rel_error < tolerance);
    if (passed) {
        std::cout << " PASS" << std::endl;
    } else {
        std::cout << " FAIL" << std::endl;
    }
    
    return passed;
}

void test_simple_addition() {
    std::cout << "\n=== Test 1: Addition ===" << std::endl;
    
    auto a = Variable::create(Tensor(1, 1), true);
    auto b = Variable::create(Tensor(1, 1), true);
    
    a->getData().setValue(0, 0, 2.0f);
    b->getData().setValue(0, 0, 3.0f);
    
    a->zeroGrad();
    auto c = a->add(b);
    c->backward();
    
    float analytical = a->getGrad().getValue(0, 0);
    
    auto forward_fn = [&]() {
        auto c_temp = a->add(b);
        return c_temp->getData().getValue(0, 0);
    };
    
    float numerical = numericalGradient(forward_fn, &a->getData().raw()[0]);
    
    bool passed = checkGradient(analytical, numerical, 2e-3);
    std::cout << (passed ? "✓ Addition gradient correct" : "✗ Addition gradient FAILED") << std::endl;
}

void test_scale() {
    std::cout << "\n=== Test 2: Scale ===" << std::endl;
    
    auto a = Variable::create(Tensor(1, 1), true);
    a->getData().setValue(0, 0, 3.0f);
    
    a->zeroGrad();
    auto c = a->scale(2.5f);
    c->backward();
    
    float analytical = a->getGrad().getValue(0, 0);
    
    auto forward_fn = [&]() {
        auto c_temp = a->scale(2.5f);
        return c_temp->getData().getValue(0, 0);
    };
    
    float numerical = numericalGradient(forward_fn, &a->getData().raw()[0]);
    
    bool passed = checkGradient(analytical, numerical, 1e-2);
    std::cout << (passed ? "✓ Scale gradient correct" : "✗ Scale gradient FAILED") << std::endl;
}

void test_matmul() {
    std::cout << "\n=== Test 3: Matrix Multiply ===" << std::endl;
    
    auto a = Variable::create(Tensor(1, 3), true);
    auto b = Variable::create(Tensor(3, 1), true);
    
    a->getData().setValue(0, 0, 1.0f);
    a->getData().setValue(0, 1, 2.0f);
    a->getData().setValue(0, 2, 3.0f);
    
    b->getData().setValue(0, 0, 0.5f);
    b->getData().setValue(1, 0, 1.5f);
    b->getData().setValue(2, 0, 2.5f);
    
    a->zeroGrad();
    b->zeroGrad();
    auto c = a->matmul(b);
    c->backward();
    
    float analytical = a->getGrad().getValue(0, 0);
    
    float epsilon = 1e-5f;
    float original = a->getData().getValue(0, 0);
    
    std::cout << "Manual numerical gradient computation:" << std::endl;
    std::cout << "Original a[0,0]: " << original << std::endl;
    
    a->getData().setValue(0, 0, original + epsilon);
    std::cout << "a[0,0] + eps: " << a->getData().getValue(0, 0) << std::endl;
    auto c_plus = a->matmul(b);
    float loss_plus = c_plus->getData().getValue(0, 0);
    std::cout << "loss_plus: " << loss_plus << std::endl;
    std::cout << "Expected: (1+" << epsilon << ")*0.5 + 2*1.5 + 3*2.5 = " << ((original+epsilon)*0.5f + 2.0f*1.5f + 3.0f*2.5f) << std::endl;
    
    a->getData().setValue(0, 0, original - epsilon);
    std::cout << "a[0,0] - eps: " << a->getData().getValue(0, 0) << std::endl;
    auto c_minus = a->matmul(b);
    float loss_minus = c_minus->getData().getValue(0, 0);
    std::cout << "loss_minus: " << loss_minus << std::endl;
    std::cout << "Expected: (1-" << epsilon << ")*0.5 + 2*1.5 + 3*2.5 = " << ((original-epsilon)*0.5f + 2.0f*1.5f + 3.0f*2.5f) << std::endl;
    
    a->getData().setValue(0, 0, original);
    
    float diff = loss_plus - loss_minus;
    float numerical = diff / (2.0f * epsilon);
    
    std::cout << "Difference: " << diff << std::endl;
    std::cout << "2*epsilon: " << (2.0f * epsilon) << std::endl;
    std::cout << "Numerical: " << numerical << std::endl;
    std::cout << "Analytical: " << analytical << std::endl;
    
    bool passed = checkGradient(analytical, numerical, 5e-2);
    std::cout << (passed ? "✓ Matmul gradient correct" : "✗ Matmul gradient FAILED") << std::endl;
}
  
void test_gelu() {
    std::cout << "\n=== Test 4: GELU ===" << std::endl;
    
    auto x = Variable::create(Tensor(1, 1), true);
    x->getData().setValue(0, 0, 0.5f);
    
    x->zeroGrad();
    auto y = x->gelu();
    y->backward();
    
    float analytical = x->getGrad().getValue(0, 0);
    
    auto forward_fn = [&]() {
        auto y_temp = x->gelu();
        return y_temp->getData().getValue(0, 0);
    };
    
    float numerical = numericalGradient(forward_fn, &x->getData().raw()[0]);
    
    bool passed = checkGradient(analytical, numerical, 1e-2);
    std::cout << (passed ? "✓ GELU gradient correct" : "✗ GELU gradient FAILED") << std::endl;
}

void test_softmax() {
    std::cout << "\n=== Test 5: Softmax (single output) ===" << std::endl;
    
    auto x = Variable::create(Tensor(1, 1), true);
    x->getData().setValue(0, 0, 1.0f);
    
    x->zeroGrad();
    auto y = x->softmax();
    y->backward();
    
    float analytical = x->getGrad().getValue(0, 0);
    
    auto forward_fn = [&]() {
        auto y_temp = x->softmax();
        return y_temp->getData().getValue(0, 0);
    };
    
    float numerical = numericalGradient(forward_fn, &x->getData().raw()[0]);
    
    bool passed = checkGradient(analytical, numerical, 1e-2);
    std::cout << (passed ? "✓ Softmax gradient correct" : "✗ Softmax gradient FAILED") << std::endl;
}

int main() {
    std::cout << "=== Gradient Checking Test Suite ===" << std::endl;
    
    test_simple_addition();
    test_scale();
    test_matmul();
    test_gelu();
    test_softmax();
    
    std::cout << "\n=== All Tests Complete ===" << std::endl;
    return 0;
}