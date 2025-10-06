#include "transformer/gpt_model.h"
#include "transformer/variable.h"
#include "transformer/optimizer.h"
#include "data/dataset.h"
#include "data/dataloader.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <limits>
#include <fstream>
#include <chrono>

void train_overfitting_test() {
    std::cout << "=== Overfitting Test: Can the model memorize a tiny sequence? ===" << std::endl;
    
    int vocab_size = 20;
    int d_model = 32;
    int num_layers = 2;
    int num_heads = 4;
    int max_len = 10;
    int seq_length = 5;
    
    std::cout << "Creating model..." << std::endl;
    GPTModel model(vocab_size, d_model, num_layers, num_heads, max_len);
    
    std::cout << "Collecting parameters..." << std::endl;
    auto params = model.getAllParameters();
    std::cout << "Total parameters: " << params.size() << std::endl;
    
    float learning_rate = 0.001f;
    AdamOptimizer optimizer(params, learning_rate);
    
    std::vector<int> tiny_sequence = {1, 2, 3, 4, 5};
    
    Tensor input_tensor(seq_length, 1);
    Tensor target_tensor(seq_length, 1);
    
    for (int i = 0; i < seq_length; i++) {
        input_tensor.setValue(i, 0, static_cast<float>(tiny_sequence[i]));
        target_tensor.setValue(i, 0, static_cast<float>(tiny_sequence[i]));
    }
    
    std::cout << "\nTarget sequence to memorize: ";
    for (int token : tiny_sequence) {
        std::cout << token << " ";
    }
    std::cout << "\n" << std::endl;
    
    std::cout << "Starting training..." << std::endl;
    std::cout << std::setw(10) << "Step" 
              << std::setw(15) << "Loss" 
              << std::setw(20) << "Grad Norm" << std::endl;
    std::cout << std::string(45, '-') << std::endl;
    
    int num_steps = 100;
    
    for (int step = 0; step < num_steps; step++) {
        auto input = Variable::create(input_tensor, false);
        auto target = Variable::create(target_tensor, false);
        
        auto logits = model.forward(input, true);
        auto log_probs = logits->log_softmax();
        auto loss = log_probs->nll_loss(target);
        
        optimizer.zero_grad();
        
        loss->backward();
        
        float grad_norm = 0.0f;
        for (const auto& param : params) {
            const Tensor& grad = param->getGrad();
            for (int i = 0; i < grad.numel(); i++) {
                float g = grad.raw()[i];
                grad_norm += g * g;
            }
        }
        grad_norm = std::sqrt(grad_norm);
        
        optimizer.clip_grad_norm(1.0f);
        
        optimizer.step();
        
        float loss_value = loss->getData().getValue(0, 0);
        
        if (step % 10 == 0 || step < 5) {
            std::cout << std::setw(10) << step 
                      << std::setw(15) << std::fixed << std::setprecision(6) << loss_value
                      << std::setw(20) << std::fixed << std::setprecision(4) << grad_norm
                      << std::endl;
        }
        
        if (step == 0) {
            if (grad_norm < 1e-6f) {
                std::cout << "\n❌ CRITICAL: Gradients are zero! No learning will occur." << std::endl;
                std::cout << "This means gradient flow is broken somewhere." << std::endl;
                return;
            } else {
                std::cout << "✓ Gradients are flowing (norm > 0)" << std::endl;
            }
        }
    }
    
    std::cout << "\n=== Final Evaluation ===" << std::endl;
    
    auto final_input = Variable::create(input_tensor, false);
    auto final_logits = model.forward(final_input, false);
    
    std::cout << "Predicted tokens: ";
    for (int i = 0; i < seq_length; i++) {
        float max_val = -1e9f;
        int max_idx = 0;
        
        for (int j = 0; j < vocab_size; j++) {
            float val = final_logits->getData().getValue(i, j);
            if (val > max_val) {
                max_val = val;
                max_idx = j;
            }
        }
        
        std::cout << max_idx << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Target tokens:    ";
    for (int token : tiny_sequence) {
        std::cout << token << " ";
    }
    std::cout << "\n" << std::endl;
    
    int correct = 0;
    for (int i = 0; i < seq_length; i++) {
        float max_val = -1e9f;
        int max_idx = 0;
        
        for (int j = 0; j < vocab_size; j++) {
            float val = final_logits->getData().getValue(i, j);
            if (val > max_val) {
                max_val = val;
                max_idx = j;
            }
        }
        
        if (max_idx == tiny_sequence[i]) {
            correct++;
        }
    }
    
    float accuracy = (float)correct / seq_length * 100.0f;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    
    if (accuracy >= 80.0f) {
        std::cout << "\n✅ SUCCESS! Model learned to memorize the sequence." << std::endl;
        std::cout << "Gradient flow is working correctly!" << std::endl;
    } else if (accuracy >= 40.0f) {
        std::cout << "\n⚠️  PARTIAL SUCCESS: Model is learning but slowly." << std::endl;
        std::cout << "Try more steps or higher learning rate." << std::endl;
    } else {
        std::cout << "\n❌ FAILURE: Model did not learn." << std::endl;
        std::cout << "Check if gradients are flowing properly." << std::endl;
    }
}

void train_with_dataloader() {
    std::cout << "\n\n=== Training with DataLoader ===" << std::endl;

    std::vector<int> tokens;
    for (int i = 0; i < 200; i++) {
        tokens.push_back(i % 15);
    }

    int vocab_size = 15;
    int d_model = 24;
    int num_layers = 2;
    int num_heads = 4;
    int max_len = 16;
    int seq_length = 8;
    int batch_size = 2;

    std::cout << "Creating model and dataset..." << std::endl;
    GPTModel model(vocab_size, d_model, num_layers, num_heads, max_len);
    auto dataset = std::make_shared<TextDataset>(tokens, seq_length);
    DataLoader loader(dataset, batch_size, true);

    auto params = model.getAllParameters();
    AdamOptimizer optimizer(params, 0.00001f);

    std::cout << "Dataset: " << dataset->size() << " sequences" << std::endl;
    std::cout << "Batches per epoch: " << loader.num_batches() << std::endl;

    int num_epochs = 3;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        std::cout << "\n--- Epoch " << epoch + 1 << " ---" << std::endl;

        loader.reset();
        float epoch_loss = 0.0f;
        int batch_count = 0;

        while (loader.has_next()) {
            auto batch = loader.next_batch();

            batch_count++;

            auto input = Variable::create(batch.input, false);
            auto target = Variable::create(batch.target, false);

            std::shared_ptr<Variable> logits;
            std::shared_ptr<Variable> probs;
            std::shared_ptr<Variable> loss;
            try {
                logits = model.forward(input, true);
                probs = logits->softmax();
                loss = probs->cross_entropy_loss(target);
            } catch (const std::exception& e) {
                std::cerr << "Error in DataLoader epoch " << epoch + 1 << ", batch " << batch_count
                          << ": " << e.what() << std::endl;
                std::cerr << "Input shape: " << batch.input.getRows() << "x" << batch.input.getCols() << std::endl;
                std::cerr << "Input is3D: " << batch.input.getIs3D() << std::endl;
                if (batch.input.getIs3D()) {
                    std::cerr << "Batch size: " << batch.input.getBatchSize() << std::endl;
                }
                std::cerr << "Target shape: " << batch.target.getRows() << "x" << batch.target.getCols() << std::endl;
                std::cerr << "Target is3D: " << batch.target.getIs3D() << std::endl;
                throw;
            }

            optimizer.zero_grad();
            loss->backward();
            optimizer.clip_grad_norm(1.0f);
            optimizer.step();

            float loss_value = loss->getData().getValue(0, 0);
            epoch_loss += loss_value;

            if (batch_count % 10 == 0) {
                std::cout << "  Batch " << batch_count << " - Loss: "
                          << std::fixed << std::setprecision(4) << loss_value << std::endl;
            }
        }

        std::cout << "Average Loss: " << (epoch_loss / batch_count) << std::endl;
    }

    std::cout << "\n✅ Multi-batch training completed!" << std::endl;
}

void train_shakespeare() {
    std::cout << "\n\n=== Shakespeare Overfitting Test (Goal 1) ===" << std::endl;
    
    // Read Shakespeare data
    std::ifstream file("data/shakespeare.txt");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open data/shakespeare.txt" << std::endl;
        std::cerr << "Run: curl https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/shakespeare.txt" << std::endl;
        return;
    }
    
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    
    std::cout << "Loaded " << text.length() << " characters" << std::endl;
    
    // Convert to byte-level tokens (simple: each char = one token)
    std::vector<int> tokens;
    for (char c : text) {
        tokens.push_back(static_cast<unsigned char>(c));
    }
    std::cout << "Created " << tokens.size() << " tokens" << std::endl;
    
    // Model config: ~10M parameters
    int vocab_size = 256;      // Byte-level vocabulary
    int d_model = 384;
    int num_layers = 6;
    int num_heads = 6;
    int max_len = 512;
    int seq_length = 256;      // Train on 256-token sequences
    int batch_size = 4;        // Process 4 sequences at once
    float learning_rate = 3e-4f;
    float dropout_rate = 0.1f;
    
    std::cout << "\nCreating model with config:" << std::endl;
    std::cout << "  vocab_size: " << vocab_size << std::endl;
    std::cout << "  d_model: " << d_model << std::endl;
    std::cout << "  num_layers: " << num_layers << std::endl;
    std::cout << "  num_heads: " << num_heads << std::endl;
    std::cout << "  seq_length: " << seq_length << std::endl;
    std::cout << "  batch_size: " << batch_size << std::endl;
    
    GPTModel model(vocab_size, d_model, num_layers, num_heads, max_len, dropout_rate);
    
    auto params = model.getAllParameters();
    int total_params = 0;
    for (const auto& p : params) {
        total_params += p->getData().numel();
    }
    std::cout << "\nTotal parameters: " << total_params / 1000000.0f << "M" << std::endl;
    
    auto dataset = std::make_shared<TextDataset>(tokens, seq_length);
    DataLoader loader(dataset, batch_size, true);
    
    AdamOptimizer optimizer(params, learning_rate);
    
    std::cout << "Dataset: " << dataset->size() << " sequences" << std::endl;
    std::cout << "Batches per epoch: " << loader.num_batches() << std::endl;
    
    int num_steps = 1000;
    int log_interval = 50;
    int save_interval = 500;
    
    std::cout << "\nTraining for " << num_steps << " steps..." << std::endl;
    std::cout << std::setw(10) << "Step" 
              << std::setw(15) << "Loss" 
              << std::setw(15) << "Grad Norm" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < num_steps; step++) {
        if (!loader.has_next()) {
            loader.reset();
        }
        
        auto batch = loader.next_batch();
        auto input = Variable::create(batch.input, false);
        auto target = Variable::create(batch.target, false);
        
        // Forward pass
        std::shared_ptr<Variable> logits;
        std::shared_ptr<Variable> log_probs;
        std::shared_ptr<Variable> loss;
        try {
            logits = model.forward(input, true);
            log_probs = logits->log_softmax();
            loss = log_probs->nll_loss(target);
        } catch (const std::exception& e) {
            std::cerr << "Error at step " << step << ": " << e.what() << std::endl;
            std::cerr << "Input shape: " << batch.input.getRows() << "x" << batch.input.getCols() << std::endl;
            std::cerr << "Input is3D: " << batch.input.getIs3D() << std::endl;
            if (batch.input.getIs3D()) {
                std::cerr << "Batch size: " << batch.input.getBatchSize() << std::endl;
            }
            std::cerr << "Target shape: " << batch.target.getRows() << "x" << batch.target.getCols() << std::endl;
            std::cerr << "Target is3D: " << batch.target.getIs3D() << std::endl;
            throw;
        }
        
        // Backward pass
        optimizer.zero_grad();
        loss->backward();
        
        // Compute gradient norm for monitoring
        float grad_norm = 0.0f;
        for (const auto& param : params) {
            const Tensor& grad = param->getGrad();
            for (int i = 0; i < grad.numel(); i++) {
                float g = grad.raw()[i];
                grad_norm += g * g;
            }
        }
        grad_norm = std::sqrt(grad_norm);
        
        optimizer.clip_grad_norm(1.0f);
        optimizer.step();
        
        float loss_val = loss->getData().getValue(0, 0);
        
        if (step % log_interval == 0) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
            
            std::cout << std::setw(10) << step 
                      << std::setw(15) << std::fixed << std::setprecision(6) << loss_val
                      << std::setw(15) << std::fixed << std::setprecision(4) << grad_norm
                      << "  (" << elapsed << "s)" << std::endl;
        }
        
        if (step > 0 && step % save_interval == 0) {
            std::string filename = "shakespeare_step" + std::to_string(step) + ".bin";
            model.save(filename);
            std::cout << "  Saved checkpoint: " << filename << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    std::cout << "\n=== Training Complete ===" << std::endl;
    std::cout << "Total time: " << total_time << " seconds" << std::endl;
    std::cout << "Steps per second: " << (float)num_steps / total_time << std::endl;
    
    std::cout << "\nSaving final model..." << std::endl;
    model.save("shakespeare_final.bin");
    
    std::cout << "\n✅ Goal 1 Complete: 10M model trained on Shakespeare!" << std::endl;
    std::cout << "Next: Verify it can generate text and check for overfitting" << std::endl;
}

int main() {
    std::cout << "🚀 TRANSFORMER TRAINING TEST 🚀\n" << std::endl;
    
    try {
        train_overfitting_test();
        train_with_dataloader();
        train_shakespeare();
        
        std::cout << "\n\n✅ ALL TRAINING TESTS COMPLETED!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "\n❌ Training failed with error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}