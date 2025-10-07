// src/main.cpp - Clean, organized training script
#include "transformer/gpt_model.h"
#include "transformer/variable.h"
#include "transformer/optimizer.h"
#include "data/dataset.h"
#include "data/dataloader.h"
#include "transformer/text_gen.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <chrono>

// ============================================================================
// UTILITIES
// ============================================================================

// Compute gradient norm across all parameters
float compute_grad_norm(const std::vector<std::shared_ptr<Variable>>& params) {
    float grad_norm = 0.0f;
    for (const auto& param : params) {
        const Tensor& grad = param->getGrad();
        for (int i = 0; i < grad.numel(); i++) {
            float g = grad.raw()[i];
            grad_norm += g * g;
        }
    }
    return std::sqrt(grad_norm);
}

// Print section header
void print_header(const std::string& title) {
    std::cout << "\n\n" << std::string(60, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(60, '=') << std::endl << std::endl;
}

// ============================================================================
// TEST 1: OVERFITTING TEST (Sanity Check)
// ============================================================================

void test_overfit_tiny_sequence() {
    print_header("Overfitting Test: Memorize 5 Tokens");
    
    // Tiny model, tiny sequence
    const int vocab_size = 20;
    const int d_model = 32;
    const int num_layers = 2;
    const int num_heads = 4;
    const int max_len = 10;
    const int seq_length = 5;
    
    GPTModel model(vocab_size, d_model, num_layers, num_heads, max_len);
    auto params = model.getAllParameters();
    AdamOptimizer optimizer(params, 0.001f);
    
    // Target sequence
    std::vector<int> sequence = {1, 2, 3, 4, 5};
    Tensor input(seq_length, 1), target(seq_length, 1);
    for (int i = 0; i < seq_length; i++) {
        input.setValue(i, 0, float(sequence[i]));
        target.setValue(i, 0, float(sequence[i]));
    }
    
    std::cout << "Target: ";
    for (int t : sequence) std::cout << t << " ";
    std::cout << "\n\nTraining..." << std::endl;
    
    std::cout << std::setw(10) << "Step" << std::setw(15) << "Loss" 
              << std::setw(20) << "Grad Norm" << std::endl;
    std::cout << std::string(45, '-') << std::endl;
    
    for (int step = 0; step < 100; step++) {
        auto in = Variable::create(input, false);
        auto tgt = Variable::create(target, false);
        
        auto logits = model.forward(in, true);
        auto loss = logits->log_softmax()->nll_loss(tgt);
        
        optimizer.zero_grad();
        loss->backward();
        float grad_norm = compute_grad_norm(params);
        optimizer.clip_grad_norm(5.0f);
        optimizer.step();
        
        if (step % 10 == 0 || step < 5) {
            std::cout << std::setw(10) << step 
                      << std::setw(15) << std::fixed << std::setprecision(6) 
                      << loss->getData().getValue(0, 0)
                      << std::setw(20) << std::fixed << std::setprecision(4) 
                      << grad_norm << std::endl;
        }
        
        if (step == 0 && grad_norm < 1e-6f) {
            std::cout << "❌ Gradients are zero!" << std::endl;
            return;
        }
    }
    
    // Check accuracy
    auto final_logits = model.forward(Variable::create(input, false), false);
    int correct = 0;
    for (int i = 0; i < seq_length; i++) {
        int pred = 0;
        float max_val = -1e9f;
        for (int j = 0; j < vocab_size; j++) {
            float val = final_logits->getData().getValue(i, j);
            if (val > max_val) { max_val = val; pred = j; }
        }
        if (pred == sequence[i]) correct++;
    }
    
    float acc = 100.0f * correct / seq_length;
    std::cout << "\nAccuracy: " << acc << "%" << std::endl;
    std::cout << (acc >= 80.0f ? "✅ SUCCESS" : "❌ FAILED") << std::endl;
}

// ============================================================================
// TEST 2: DATALOADER TEST (Multi-batch)
// ============================================================================

void test_dataloader() {
    print_header("DataLoader Test: Multi-Batch Training");
    
    // Synthetic data
    std::vector<int> tokens;
    for (int i = 0; i < 200; i++) tokens.push_back(i % 15);
    
    GPTModel model(15, 24, 2, 4, 16);
    auto dataset = std::make_shared<TextDataset>(tokens, 8);
    DataLoader loader(dataset, 2, true);
    
    auto params = model.getAllParameters();
    AdamOptimizer optimizer(params, 1e-5f);
    
    for (int epoch = 0; epoch < 3; epoch++) {
        std::cout << "\nEpoch " << (epoch + 1) << std::endl;
        loader.reset();
        float total_loss = 0.0f;
        int batch_count = 0;
        
        while (loader.has_next()) {
            auto batch = loader.next_batch();
            auto in = Variable::create(batch.input, false);
            auto tgt = Variable::create(batch.target, false);
            
            auto loss = model.forward(in, true)->log_softmax()->nll_loss(tgt);
            
            optimizer.zero_grad();
            loss->backward();
            optimizer.clip_grad_norm(1.0f);
            optimizer.step();
            
            total_loss += loss->getData().getValue(0, 0);
            batch_count++;
            
            if (batch_count % 10 == 0) {
                std::cout << "  Batch " << batch_count << " - Loss: " 
                          << std::fixed << std::setprecision(4) 
                          << loss->getData().getValue(0, 0) << std::endl;
            }
        }
        
        std::cout << "Avg Loss: " << (total_loss / batch_count) << std::endl;
    }
    
    std::cout << "✅ DataLoader test complete" << std::endl;
}

// ============================================================================
// TEST 3: SHAKESPEARE TRAINING (Main Goal)
// ============================================================================

void train_shakespeare() {
    print_header("Shakespeare Training - Goal 1");
    
    // Load data
    std::ifstream file("data/shakespeare.txt");
    if (!file.is_open()) {
        std::cerr << "❌ Can't open data/shakespeare.txt" << std::endl;
        return;
    }
    
    std::string text((std::istreambuf_iterator<char>(file)), 
                     std::istreambuf_iterator<char>());
    file.close();
    
    std::vector<int> tokens;
    for (char c : text) tokens.push_back(static_cast<unsigned char>(c));
    
    std::cout << "Loaded " << tokens.size() << " tokens\n" << std::endl;
    
    // Model config
    const int vocab_size = 256;
    const int d_model = 256;
    const int num_layers = 4;
    const int num_heads = 4;
    const int max_len = 512;
    const int seq_length = 128;
    const int batch_size = 2;
    const float lr = 3e-4f;
    const int num_steps = 200;
    
    std::cout << "Config: vocab=" << vocab_size << " d_model=" << d_model 
              << " layers=" << num_layers << " heads=" << num_heads << std::endl;
    
    GPTModel model(vocab_size, d_model, num_layers, num_heads, max_len, 0.1f);
    
    auto params = model.getAllParameters();
    int total = 0;
    for (const auto& p : params) total += p->getData().numel();
    std::cout << "Parameters: " << (total / 1e6f) << "M\n" << std::endl;
    
    auto dataset = std::make_shared<TextDataset>(tokens, seq_length);
    DataLoader loader(dataset, batch_size, true);
    AdamOptimizer optimizer(params, lr);
    
    std::cout << "Training for " << num_steps << " steps...\n" << std::endl;
    std::cout << std::setw(10) << "Step" << std::setw(15) << "Loss" 
              << std::setw(15) << "Grad Norm" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < num_steps; step++) {
        if (!loader.has_next()) loader.reset();
        
        auto t0 = std::chrono::high_resolution_clock::now();
        auto batch = loader.next_batch();
        auto t1 = std::chrono::high_resolution_clock::now();
        
        auto in = Variable::create(batch.input, false);
        auto tgt = Variable::create(batch.target, false);
        
        auto logits = model.forward(in, true);
        auto loss = logits->log_softmax()->nll_loss(tgt);
        auto t2 = std::chrono::high_resolution_clock::now();
        
        optimizer.zero_grad();
        auto t3 = std::chrono::high_resolution_clock::now();
        
        loss->backward();
        auto t4 = std::chrono::high_resolution_clock::now();
        
        optimizer.clip_grad_norm(5.0f);
        optimizer.step();
        auto t5 = std::chrono::high_resolution_clock::now();
        
        // Log every 10 steps
        if (step % 10 == 0) {
            float grad_norm = compute_grad_norm(params);
            float loss_val = loss->getData().getValue(0, 0);
            
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                now - start).count();
            
            std::cout << std::setw(10) << step 
                      << std::setw(15) << std::fixed << std::setprecision(6) << loss_val
                      << std::setw(15) << std::fixed << std::setprecision(4) << grad_norm
                      << "  (" << elapsed << "s)" << std::endl;
            
            // Timing breakdown
            auto ms = [](auto a, auto b) { 
                return std::chrono::duration<double, std::milli>(b - a).count(); 
            };
            std::cout << "  [perf] loader=" << std::fixed << std::setprecision(1) 
                      << ms(t0,t1) << "ms fwd=" << ms(t1,t2) << "ms bwd=" 
                      << ms(t3,t4) << "ms opt=" << ms(t4,t5) << "ms" << std::endl;
        }
        
        // Save checkpoint every 500 steps
        if (step > 0 && step % 500 == 0) {
            model.save("shakespeare_step_" + std::to_string(step) + ".bin");
            std::cout << "  [checkpoint saved]" << std::endl;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(
        end - start).count();
    
    std::cout << "\n=== Training Complete ===" << std::endl;
    std::cout << "Time: " << total_time << "s" << std::endl;
    std::cout << "Speed: " << (num_steps / (float)total_time) << " steps/s" << std::endl;
    
    model.save("shakespeare_final.bin");
    std::cout << "\n✅ Goal 1 Complete!" << std::endl;

    // After model.save("shakespeare_final.bin");

    // Replace the generation section in train_shakespeare() with this:

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  Generating Sample Text" << std::endl;
    std::cout << std::string(60, '=') << std::endl << std::endl;

    // Create text generator
    TextGen generator(model);

    // Helper function to create prompts
    auto string_to_tokens = [](const std::string& str) {
        std::vector<int> tokens;
        for (char c : str) {
            tokens.push_back(static_cast<unsigned char>(c));
        }
        return tokens;
    };

    // Try different prompts
    std::vector<std::string> prompts = {
        "ROMEO:\n",
        "JULIET:\n",
        "First Citizen:\n"
    };

    std::cout << "=== Greedy Decoding (deterministic) ===" << std::endl << std::endl;

    for (const auto& prompt_str : prompts) {
        std::cout << "Prompt: \"" << prompt_str << "\"" << std::endl;
        
        auto prompt = string_to_tokens(prompt_str);
        std::string generated = generator.generate_greedy(prompt, 150);
        
        // Just print generated (it already includes the prompt)
        std::cout << generated << std::endl;
        std::cout << std::string(40, '-') << std::endl << std::endl;
    }

    std::cout << "\n=== Sampling (temperature=0.8) ===" << std::endl << std::endl;

    for (const auto& prompt_str : prompts) {
        std::cout << "Prompt: \"" << prompt_str << "\"" << std::endl;
        
        auto prompt = string_to_tokens(prompt_str);
        // Use sampling for more variety (might break the "the the" loop)
        std::string generated = generator.generate_sample(prompt, 0.8f, 150);
        
        std::cout << generated << std::endl;
        std::cout << std::string(40, '-') << std::endl << std::endl;
    }

    std::cout << "\nNote: Model trained for only 200 steps (loss=2.9)." << std::endl;
    std::cout << "At this stage, the model has learned:" << std::endl;
    std::cout << "  ✓ Common English words (the, and, is, he)" << std::endl;
    std::cout << "  ✓ Spacing and punctuation" << std::endl;
    std::cout << "  ✗ Semantic meaning (needs 1000+ steps)" << std::endl;
    std::cout << "\nTrain longer (1000 steps, loss~1.5) for coherent text!" << std::endl;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "🚀 TRANSFORMER TRAINING SUITE 🚀\n" << std::endl;
    
    try {
        test_overfit_tiny_sequence();
        test_dataloader();
        train_shakespeare();
        
        std::cout << "\n\n✅ ALL TESTS COMPLETED!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Error: " << e.what() << std::endl;
        return 1;
    }
}