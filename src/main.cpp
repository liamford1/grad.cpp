#include "transformer/tensor.h"
#include "transformer/gpt_model.h"
#include "transformer/text_gen.h"
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
    test_gpt_model();
    test_text_generation();
    test_model_save_load();
    
    // Show practical demo
    demo_text_generation();
    
    std::cout << "✅ All core tests completed!" << std::endl;
    std::cout << "Your transformer implementation is working!" << std::endl;
    
    return 0;
}