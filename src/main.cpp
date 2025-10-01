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
        auto probs = logits->softmax();
        auto loss = probs->cross_entropy_loss(target);
        
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
    int batch_size = 1;

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

            auto logits = model.forward(input, true);
            auto probs = logits->softmax();
            auto loss = probs->cross_entropy_loss(target);

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

int main() {
    std::cout << "🚀 TRANSFORMER TRAINING TEST 🚀\n" << std::endl;
    
    try {
        train_overfitting_test();
        
        train_with_dataloader();
        
        std::cout << "\n\n✅ ALL TRAINING TESTS COMPLETED!" << std::endl;
        std::cout << "If loss decreased, your transformer can learn!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "\n❌ Training failed with error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}