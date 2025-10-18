#include <iostream>
#include <cmath>
#include "transformer/variable.h"
#include "transformer/tensor.h"

void test_dropout_training_mode() {
    std::cout << "\n=== Test 1: Dropout in Training Mode ===" << std::endl;
    
    auto x = Variable::create(Tensor(1, 100), true);
    for (int i = 0; i < 100; i++) {
        x->getData().raw()[i] = 1.0f;
    }
    
    float dropout_rate = 0.5f;
    auto y = x->dropout(dropout_rate, true);
    
    int num_zeros = 0;
    int num_scaled = 0;
    float expected_scale = 1.0f / (1.0f - dropout_rate);
    
    for (int i = 0; i < 100; i++) {
        float val = y->getData().raw()[i];
        if (val == 0.0f) {
            num_zeros++;
        } else if (std::abs(val - expected_scale) < 0.001f) {
            num_scaled++;
        }
    }
    
    std::cout << "Dropout rate: " << dropout_rate << std::endl;
    std::cout << "Zeros: " << num_zeros << "/100 (expected ~50)" << std::endl;
    std::cout << "Scaled values: " << num_scaled << "/100" << std::endl;
    std::cout << "Expected scale: " << expected_scale << std::endl;
    
    bool reasonable_dropout = (num_zeros >= 35 && num_zeros <= 65);
    std::cout << (reasonable_dropout ? "✓ Dropout rate looks correct" : "✗ Dropout rate suspicious") << std::endl;
}

void test_dropout_inference_mode() {
    std::cout << "\n=== Test 2: Dropout in Inference Mode ===" << std::endl;
    
    auto x = Variable::create(Tensor(1, 10), true);
    for (int i = 0; i < 10; i++) {
        x->getData().raw()[i] = float(i + 1);
    }
    
    auto y = x->dropout(0.5f, false);
    
    bool all_unchanged = true;
    for (int i = 0; i < 10; i++) {
        if (y->getData().raw()[i] != x->getData().raw()[i]) {
            all_unchanged = false;
            break;
        }
    }
    
    std::cout << (all_unchanged ? "✓ Inference mode unchanged" : "✗ Inference mode modified") << std::endl;
}

void test_dropout_mean_preservation() {
    std::cout << "\n=== Test 3: Mean Preservation ===" << std::endl;
    
    auto x = Variable::create(Tensor(1, 1000), true);
    for (int i = 0; i < 1000; i++) {
        x->getData().raw()[i] = 5.0f;
    }
    
    float dropout_rate = 0.3f;
    auto y = x->dropout(dropout_rate, true);
    
    float sum = 0.0f;
    for (int i = 0; i < 1000; i++) {
        sum += y->getData().raw()[i];
    }
    float mean = sum / 1000.0f;
    
    std::cout << "Input mean: 5.0" << std::endl;
    std::cout << "Output mean: " << mean << std::endl;
    std::cout << "Expected: ~5.0 (due to scaling)" << std::endl;
    
    bool mean_preserved = (std::abs(mean - 5.0f) < 0.5f);
    std::cout << (mean_preserved ? "✓ Mean preserved" : "✗ Mean not preserved") << std::endl;
}

int main() {
    std::cout << "=== Dropout Verification Tests ===" << std::endl;
    
    test_dropout_training_mode();
    test_dropout_inference_mode();
    test_dropout_mean_preservation();
    
    std::cout << "\n=== Dropout Tests Complete ===" << std::endl;
    return 0;
}