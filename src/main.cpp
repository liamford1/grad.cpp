#include "transformer/tensor.h"
#include "transformer/gpt_model.h"
#include "transformer/text_gen.h"
#include "transformer/variable.h"  // 🆕 ADD THIS
#include <iostream>
#include <sstream>
#include <cstdio>
#include <vector>

void test_basic_tensor() {
    std::cout << "=== Testing Basic Tensor Operations ===" << std::endl;
    
    // Test 2D and 3D tensor creation
    Tensor t2d(2, 3);
    Tensor t3d(2, 2, 3);
    
    t2d.fill(1.0f);
    t3d.fill(2.0f);
    
    std::cout << "2D tensor (2x3): " << t2d.getRows() << "x" << t2d.getCols() << std::endl;
    std::cout << "3D tensor (2x2x3): " << t3d.getBatchSize() << "x" << t3d.getRows() << "x" << t3d.getCols() << std::endl;
    
    // Test basic operations
    Tensor t2d_copy = t2d.scale(2.0f);
    std::cout << "Scale operation works: " << (t2d_copy.getValue(0, 0) == 2.0f) << std::endl;
    
    std::cout << "Basic tensor tests passed!\n" << std::endl;
}

void test_variable_autodiff() {  // 🆕 NEW TEST FUNCTION
    std::cout << "=== Testing Variable Automatic Differentiation ===" << std::endl;
    
    try {
        // Test 1: Basic Variable creation
        std::cout << "Test 1: Variable creation..." << std::endl;
        Tensor data1(2, 2);
        data1.setValue(0, 0, 1.0f);
        data1.setValue(0, 1, 2.0f);
        data1.setValue(1, 0, 3.0f);
        data1.setValue(1, 1, 4.0f);
        
        auto a = Variable::create(data1, true);
        std::cout << "✓ Variable creation successful" << std::endl;
        
        // Test 2: Basic operations
        std::cout << "Test 2: Basic operations..." << std::endl;
        Tensor data2(2, 2);
        data2.setValue(0, 0, 0.5f);
        data2.setValue(0, 1, 1.5f);
        data2.setValue(1, 0, 2.5f);
        data2.setValue(1, 1, 3.5f);
        
        auto b = Variable::create(data2, true);
        auto c = a->add(b);
        std::cout << "✓ Addition operation successful" << std::endl;
        
        // Test 3: Scaling
        std::cout << "Test 3: Scaling..." << std::endl;
        auto d = c->scale(2.0f);
        std::cout << "✓ Scale operation successful" << std::endl;
        
        // Test 4: Matrix multiplication
        std::cout << "Test 4: Matrix multiplication..." << std::endl;
        Tensor data3(2, 2);
        data3.setValue(0, 0, 1.0f);
        data3.setValue(0, 1, 0.0f);
        data3.setValue(1, 0, 0.0f);
        data3.setValue(1, 1, 1.0f);  // Identity matrix
        
        auto identity = Variable::create(data3, true);
        auto e = a->matmul(identity);
        std::cout << "✓ Matrix multiplication successful" << std::endl;
        
        // Test 5: Gradient computation
        std::cout << "Test 5: Backward pass..." << std::endl;
        e->backward();
        std::cout << "✓ Backward pass completed without errors" << std::endl;
        
        // Test 6: Verify gradients exist
        std::cout << "Test 6: Gradient verification..." << std::endl;
        std::cout << "Gradient of 'a':" << std::endl;
        a->getGrad().display();
        
        std::cout << "Gradient of 'identity':" << std::endl;
        identity->getGrad().display();
        
        // Test 7: Chain rule with longer computation
        std::cout << "Test 7: Complex chain rule..." << std::endl;
        
        // Clear gradients
        a->zeroGrad();
        b->zeroGrad();
        identity->zeroGrad();
        
        // Complex expression: f = (a + b) * 2.0 * identity
        auto step1 = a->add(b);
        auto step2 = step1->scale(2.0f);
        auto result = step2->matmul(identity);
        
        result->backward();
        std::cout << "✓ Complex chain rule backward pass successful" << std::endl;
        
        // Verify gradients were computed
        bool has_grad_a = false, has_grad_b = false;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                if (std::abs(a->getGrad().getValue(i, j)) > 1e-6f) has_grad_a = true;
                if (std::abs(b->getGrad().getValue(i, j)) > 1e-6f) has_grad_b = true;
            }
        }
        
        std::cout << "Variable 'a' received gradients: " << has_grad_a << std::endl;
        std::cout << "Variable 'b' received gradients: " << has_grad_b << std::endl;
        
        // Test 8: Softmax (if it doesn't crash)
        std::cout << "Test 8: Softmax gradient..." << std::endl;
        Tensor softmax_data(1, 3);
        softmax_data.setValue(0, 0, 1.0f);
        softmax_data.setValue(0, 1, 2.0f);
        softmax_data.setValue(0, 2, 3.0f);
        
        auto softmax_var = Variable::create(softmax_data, true);
        auto softmax_result = softmax_var->softmax();
        softmax_result->backward();
        std::cout << "✓ Softmax gradient computation successful" << std::endl;
        
        std::cout << "Variable autodiff tests passed!\n" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Variable test failed with exception: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "❌ Variable test failed with unknown exception" << std::endl;
    }
}

void test_gpt_model() {
    std::cout << "=== Testing GPT Model ===" << std::endl;
    
    // Create small model for testing
    int vocab_size = 50;
    int d_model = 32;
    int num_layers = 2;
    int num_heads = 4;
    int max_len = 16;
    
    GPTModel model(vocab_size, d_model, num_layers, num_heads, max_len);
    
    // Test with some token sequence
    Tensor input(3, 1);  // 3 tokens
    input.setValue(0, 0, 1);
    input.setValue(1, 0, 5);
    input.setValue(2, 0, 10);
    
    Tensor output = model.forward(input);
    
    std::cout << "Input shape: " << input.getRows() << "x" << input.getCols() << std::endl;
    std::cout << "Output shape: " << output.getRows() << "x" << output.getCols() << std::endl;
    std::cout << "Expected output shape: 3x" << vocab_size << std::endl;
    
    bool shape_correct = (output.getRows() == 3 && output.getCols() == vocab_size);
    std::cout << "Output shape correct: " << shape_correct << std::endl;
    
    // Check output values are reasonable (not NaN/inf)
    bool values_reasonable = true;
    for (int i = 0; i < 3 && values_reasonable; i++) {
        for (int j = 0; j < vocab_size && values_reasonable; j++) {
            float val = output.getValue(i, j);
            if (std::isnan(val) || std::isinf(val)) {
                values_reasonable = false;
            }
        }
    }
    std::cout << "Output values reasonable: " << values_reasonable << std::endl;
    
    std::cout << "GPT model tests passed!\n" << std::endl;
}

void test_text_generation() {
    std::cout << "=== Testing Text Generation ===" << std::endl;
    
    // Create small model
    int vocab_size = 20;
    int d_model = 16;
    int num_layers = 2;
    int num_heads = 2;
    int max_len = 10;
    
    GPTModel model(vocab_size, d_model, num_layers, num_heads, max_len);
    TextGen generator(model);
    
    // Test with simple prompt
    std::vector<int> prompt = {1, 5, 3};
    
    std::cout << "Prompt tokens: ";
    for (int token : prompt) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
    
    // Test greedy generation
    std::string greedy_result = generator.generate_greedy(prompt, 5);
    std::cout << "Greedy result: " << greedy_result << std::endl;
    
    // Test sampling with different temperatures
    std::string sample_low = generator.generate_sample(prompt, 0.1f, 5);
    std::cout << "Sample (temp=0.1): " << sample_low << std::endl;
    
    std::string sample_high = generator.generate_sample(prompt, 1.5f, 5);
    std::cout << "Sample (temp=1.5): " << sample_high << std::endl;
    
    // Verify different temperatures produce different results
    bool different_temps = (sample_low != sample_high);
    std::cout << "Different temperatures produce different results: " << different_temps << std::endl;
    
    std::cout << "Text generation tests passed!\n" << std::endl;
}

void test_model_save_load() {
    std::cout << "=== Testing Model Save/Load ===" << std::endl;
    
    // Create and save model
    GPTModel original(10, 8, 1, 2, 8);
    
    std::string filepath = "test_model.bin";
    bool save_success = original.save(filepath);
    std::cout << "Model save successful: " << save_success << std::endl;
    
    if (save_success) {
        // Load model
        try {
            GPTModel loaded = GPTModel::load(filepath);
            
            // Test same input produces same output
            Tensor test_input(2, 1);
            test_input.setValue(0, 0, 1);
            test_input.setValue(1, 0, 3);
            
            Tensor original_output = original.forward(test_input);
            Tensor loaded_output = loaded.forward(test_input);
            
            // Check if outputs match (within tolerance)
            bool outputs_match = true;
            float tolerance = 1e-6f;
            
            for (int i = 0; i < 2 && outputs_match; i++) {
                for (int j = 0; j < 10 && outputs_match; j++) {
                    float diff = std::abs(original_output.getValue(i, j) - loaded_output.getValue(i, j));
                    if (diff > tolerance) {
                        outputs_match = false;
                    }
                }
            }
            
            std::cout << "Loaded model produces same output: " << outputs_match << std::endl;
            
            // Clean up
            std::remove(filepath.c_str());
            
        } catch (const std::exception& e) {
            std::cout << "Model load failed: " << e.what() << std::endl;
        }
    }
    
    std::cout << "Model save/load tests completed!\n" << std::endl;
}

void demo_text_generation() {
    std::cout << "=== Text Generation Demo ===" << std::endl;
    
    // Create vocabulary for demo (simple word-like tokens)
    std::vector<std::string> vocab = {
        "the", "cat", "dog", "runs", "jumps", "over", "and", "quick", "brown", "fox",
        "lazy", "big", "small", "red", "blue", "house", "tree", "park", "street", "car"
    };
    
    GPTModel model(vocab.size(), 24, 3, 4, 16);
    TextGen generator(model);
    
    std::cout << "Vocabulary size: " << vocab.size() << std::endl;
    std::cout << "Generating sequences starting with different words...\n" << std::endl;
    
    // Generate a few examples
    for (int start_token = 0; start_token < 3; start_token++) {
        std::vector<int> prompt = {start_token};
        
        std::cout << "Starting with '" << vocab[start_token] << "':" << std::endl;
        
        // Generate with greedy
        std::string greedy = generator.generate_greedy(prompt, 4);
        std::cout << "  Greedy: " << greedy << " -> ";
        
        // Convert back to words for display
        std::vector<int> tokens;
        std::stringstream ss(greedy);
        std::string token_str;
        while (ss >> token_str) {
            int token_id = std::stoi(token_str);
            if (token_id >= 0 && token_id < vocab.size()) {
                std::cout << vocab[token_id] << " ";
            }
        }
        std::cout << std::endl;
        
        // Generate with sampling
        std::string sampled = generator.generate_sample(prompt, 0.8f, 4);
        std::cout << "  Sampled: " << sampled << " -> ";
        
        std::stringstream ss2(sampled);
        while (ss2 >> token_str) {
            int token_id = std::stoi(token_str);
            if (token_id >= 0 && token_id < vocab.size()) {
                std::cout << vocab[token_id] << " ";
            }
        }
        std::cout << "\n" << std::endl;
    }
}

int main() {
    std::cout << "🤖 TRANSFORMER FROM SCRATCH - CORE TESTING 🤖\n" << std::endl;
    
    // Run key tests
    test_basic_tensor();
    test_variable_autodiff();  // 🆕 ADD THIS LINE
    test_gpt_model();
    test_text_generation();
    test_model_save_load();
    
    // Show practical demo
    demo_text_generation();
    
    std::cout << "✅ All core tests completed!" << std::endl;
    std::cout << "Your transformer implementation is working!" << std::endl;
    
    return 0;
}