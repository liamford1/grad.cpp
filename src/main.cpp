// src/main.cpp - Clean, organized training script
#include "transformer/gpt_model.h"
#include "transformer/variable.h"
#include "transformer/optimizer.h"
#include "data/dataset.h"
#include "data/dataloader.h"
#include "transformer/text_gen.h"
#include "tokenizer/bpe_tokenizer.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <sstream>
#include <chrono>
#include <mach/mach.h>
// ============================================================================
// UTILITIES
// ============================================================================

// Compute gradient norm across all parameters
float compute_grad_norm(const std::vector<std::shared_ptr<Variable>>& params) {
    float grad_norm = 0.0f;
    for (const auto& param : params) {
        const Tensor& grad = param->getGrad();
        for (size_t i = 0; i < grad.numel(); i++) {
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

size_t get_memory_mb() {
    struct task_basic_info info;
    mach_msg_type_number_t size = sizeof(info);
    kern_return_t kerr = task_info(mach_task_self(),
                                    TASK_BASIC_INFO,
                                    (task_info_t)&info,
                                    &size);
    return (kerr == KERN_SUCCESS) ? info.resident_size / (1024 * 1024) : 0;
}

// ============================================================================
// BENCHMARK: Pure Training Performance (No I/O)
// ============================================================================

void benchmark_training_speed() {
    print_header("Performance Benchmark: 100 Steps");
    
    // Load data
    std::ifstream file("data/shakespeare.txt");
    if (!file.is_open()) {
        std::cerr << "❌ Can't open data/shakespeare.txt" << std::endl;
        return;
    }
    
    std::string text((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
    file.close();

    std::istringstream sample_stream(text);
    std::string sample_text;
    std::string word;
    int word_count = 0;
    int max_training_words = 20000;

    while (sample_stream >> word && word_count < max_training_words) {
        sample_text += word + " ";
        word_count++;
    }

    std::cout << "Training BPE tokenizer on " << word_count << " words (sampled)..." << std::endl;
    BPETokenizer tokenizer(3000);
    tokenizer.train(sample_text);
    std::cout << "Tokenizer trained! Encoding full text..." << std::endl;
    std::vector<int> tokens = tokenizer.encode(text);
    std::cout << "Encoded " << tokens.size() << " tokens." << std::endl;
    
    // Same config as your real training
    const int vocab_size = tokenizer.getCurrentVocabSize();
    const int d_model = 256;
    const int num_layers = 4;
    const int num_heads = 4;
    const int max_len = 512;
    const int seq_length = 128;
    const int batch_size = 2;
    
    GPTModel model(vocab_size, d_model, num_layers, num_heads, max_len, 0.1f);
    auto params = model.getAllParameters();
    auto dataset = std::make_shared<TextDataset>(tokens, seq_length);
    DataLoader loader(dataset, batch_size, true);
    AdamOptimizer optimizer(params, 1e-4f);
    
    std::cout << "Config: vocab=" << vocab_size << " d_model=" << d_model 
              << " layers=" << num_layers << " heads=" << num_heads << std::endl;
    std::cout << "Running 100 steps...\n" << std::endl;
    
    // Warmup (5 steps to stabilize caches)
    for (int step = 0; step < 5; step++) {
        if (!loader.has_next()) loader.reset();
        auto batch = loader.next_batch();
        
        int bs = batch.input.getBatchSize();
        int sl = batch.input.getRows();
        Tensor input_2d(bs * sl, 1), target_2d(bs * sl, 1);
        
        for (int b = 0; b < bs; b++) {
            for (int s = 0; s < sl; s++) {
                input_2d.setValue(b * sl + s, 0, batch.input.getValue(b, s, 0));
                target_2d.setValue(b * sl + s, 0, batch.target.getValue(b, s, 0));
            }
        }
        
        auto in = Variable::create(input_2d, false);
        auto tgt = Variable::create(target_2d, false);
        auto loss = model.forward(in, true)->log_softmax()->nll_loss(tgt);
        optimizer.zero_grad();
        loss->backward();
        loss->release_graph();
        optimizer.clip_grad_norm(5.0f);
        optimizer.step();
    }
    
    std::cout << "Warmup complete. Starting benchmark..." << std::endl;
    
    // BENCHMARK: 100 steps
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < 100; step++) {
        if (!loader.has_next()) loader.reset();
        auto batch = loader.next_batch();
        
        int bs = batch.input.getBatchSize();
        int sl = batch.input.getRows();
        Tensor input_2d(bs * sl, 1), target_2d(bs * sl, 1);
        
        for (int b = 0; b < bs; b++) {
            for (int s = 0; s < sl; s++) {
                input_2d.setValue(b * sl + s, 0, batch.input.getValue(b, s, 0));
                target_2d.setValue(b * sl + s, 0, batch.target.getValue(b, s, 0));
            }
        }
        
        auto in = Variable::create(input_2d, false);
        auto tgt = Variable::create(target_2d, false);
        auto loss = model.forward(in, true)->log_softmax()->nll_loss(tgt);
        optimizer.zero_grad();
        loss->backward();
        loss->release_graph();
        optimizer.clip_grad_norm(5.0f);
        optimizer.step();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "\n=== BASELINE RESULTS ===" << std::endl;
    std::cout << "100 steps took: " << duration.count() << "ms" << std::endl;
    std::cout << "Average per step: " << (duration.count() / 100.0) << "ms" << std::endl;
    std::cout << "Speed: " << (100000.0 / duration.count()) << " steps/sec" << std::endl;
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
        optimizer.clip_grad_norm(1.0f);
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
            
            // Convert 3D batch tensors to 2D for model compatibility
            // batch.input is (batch_size, seq_len, 1), we need (batch_size * seq_len, 1)
            int batch_size = batch.input.getBatchSize();
            int seq_len = batch.input.getRows();
            
            Tensor input_2d(batch_size * seq_len, 1);
            Tensor target_2d(batch_size * seq_len, 1);
            
            for (int b = 0; b < batch_size; b++) {
                for (int s = 0; s < seq_len; s++) {
                    input_2d.setValue(b * seq_len + s, 0, batch.input.getValue(b, s, 0));
                    target_2d.setValue(b * seq_len + s, 0, batch.target.getValue(b, s, 0));
                }
            }
            
            auto in = Variable::create(input_2d, false);
            auto tgt = Variable::create(target_2d, false);
            
            auto loss = model.forward(in, true)->log_softmax()->nll_loss(tgt);
            
            optimizer.zero_grad();
            loss->backward();
            loss->release_graph();
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
    print_header("Shakespeare Training... ");

    const bool FAST_MODE = true;
    const int vocab_target = FAST_MODE ? 500 : 5000;
    const int train_steps = FAST_MODE ? 50 : 1000;
    const int sequence_len = FAST_MODE ? 64 : 128;
    
    std::ifstream file("data/shakespeare.txt");
    if (!file.is_open()) {
        std::cerr << "❌ Can't open data/shakespeare.txt" << std::endl;
        return;
    }
    
    std::string text((std::istreambuf_iterator<char>(file)), 
                     std::istreambuf_iterator<char>());
    file.close();

    std::cout << "Training tokenizer..." << std::endl;
    BPETokenizer tokenizer(vocab_target);
    tokenizer.train(text);
    std::cout << "Tokenizer trained. Vocab size: " << tokenizer.getCurrentVocabSize() << std::endl;
    
    std::vector<int> tokens = tokenizer.encode(text);
    std::cout << "Loaded " << tokens.size() << " tokens\n" << std::endl;

    const int vocab_size = tokenizer.getCurrentVocabSize();
    const int d_model = 256;
    const int num_layers = 4;
    const int num_heads = 4;
    const int max_len = 512;
    const int seq_length = sequence_len;
    const int batch_size = 2;
    const float lr = 1e-4f;
    const int num_steps = train_steps;
    
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
        
        int batch_size = batch.input.getBatchSize();
        int seq_len = batch.input.getRows();

        Tensor input_2d(batch_size * seq_len, 1);
        Tensor target_2d(batch_size * seq_len, 1);
        
        for (int b = 0; b < batch_size; b++) {
            for (int s = 0; s < seq_len; s++) {
                input_2d.setValue(b * seq_len + s, 0, batch.input.getValue(b, s, 0));
                target_2d.setValue(b * seq_len + s, 0, batch.target.getValue(b, s, 0));
            }
        }
        
        auto in = Variable::create(input_2d, false);
        auto tgt = Variable::create(target_2d, false);

        auto logits = model.forward(in, true);
        auto loss = logits->log_softmax()->nll_loss(tgt);
        auto t2 = std::chrono::high_resolution_clock::now();
        
        optimizer.zero_grad();
        auto t3 = std::chrono::high_resolution_clock::now();
        
        loss->backward();
        loss->release_graph();
        auto t4 = std::chrono::high_resolution_clock::now();
        
        optimizer.clip_grad_norm(5.0f);
        optimizer.step();
        auto t5 = std::chrono::high_resolution_clock::now();
        
        float loss_val = loss->getData().getValue(0, 0);

        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
        float progress = 100.0f * (step + 1) / num_steps;
        float steps_per_sec = (step + 1) / (float)(elapsed + 1);
        int eta_sec = (int)((num_steps - step - 1) / (steps_per_sec + 0.0001f));

        std::cout << "\r[" << (step + 1) << "/" << num_steps << "] "
                  << std::fixed << std::setprecision(1) << progress << "% "
                  << "loss=" << std::setprecision(4) << loss_val << " "
                  << "speed=" << std::setprecision(2) << steps_per_sec << "it/s "
                  << "eta=" << (eta_sec / 60) << "m" << (eta_sec % 60) << "s     " << std::flush;

        if (step % 10 == 0) {
            float grad_norm = compute_grad_norm(params);

            std::cout << "\n" << std::setw(10) << step
                      << std::setw(15) << std::fixed << std::setprecision(6) << loss_val
                      << std::setw(15) << std::fixed << std::setprecision(4) << grad_norm
                      << "  (" << elapsed << "s)"
                      << " [" << get_memory_mb() << "MB]"
                      << std::endl;

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
    TextGen generator(model, &tokenizer);

    // Helper function to create prompts
    auto string_to_tokens = [&tokenizer](const std::string& str) {
        return tokenizer.encode(str);
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