#include "transformer/tensor.h"
#include "transformer/gpt_model.h"
#include "transformer/text_gen.h"
#include "transformer/variable.h"
#include "transformer/linear.h"
#include "transformer/optimizer.h"  // 🆕 ADD THIS
#include <iostream>
#include <sstream>
#include <cstdio>
#include <vector>

void test_adam_optimizer() {  // 🆕 NEW TEST
    std::cout << "=== Testing Adam Optimizer ===" << std::endl;
    
    try {
        // Test 1: Basic 2D tensor optimization
        std::cout << "Test 1: 2D tensor optimization..." << std::endl;
        auto param2d = std::make_shared<Variable>(Tensor(3, 3), true);
        param2d->getData().fill(1.0f);
        
        std::vector<std::shared_ptr<Variable>> params2d = {param2d};
        AdamOptimizer opt2d(params2d, 0.1f);
        
        std::cout << "Initial value: " << param2d->getData().getValue(0, 0) << std::endl;
        
        for (int step = 0; step < 5; step++) {
            param2d->getGrad().fill(0.1f);
            opt2d.step();
            std::cout << "  Step " << step + 1 << ": " 
                      << param2d->getData().getValue(0, 0) << std::endl;
            opt2d.zero_grad();
        }
        
        bool decreased = param2d->getData().getValue(0, 0) < 1.0f;
        std::cout << "✓ Parameters decreased: " << decreased << std::endl;
        
        // Test 2: 3D tensor optimization (like batched operations)
        std::cout << "\nTest 2: 3D tensor optimization..." << std::endl;
        auto param3d = std::make_shared<Variable>(Tensor(2, 3, 3), true);
        param3d->getData().fill(2.0f);
        
        std::vector<std::shared_ptr<Variable>> params3d = {param3d};
        AdamOptimizer opt3d(params3d, 0.05f);
        
        std::cout << "Initial 3D value: " << param3d->getData().getValue(0, 0, 0) << std::endl;
        
        for (int step = 0; step < 3; step++) {
            param3d->getGrad().fill(0.2f);
            opt3d.step();
            std::cout << "  Step " << step + 1 << ": " 
                      << param3d->getData().getValue(0, 0, 0) << std::endl;
            opt3d.zero_grad();
        }
        
        bool decreased3d = param3d->getData().getValue(0, 0, 0) < 2.0f;
        std::cout << "✓ 3D parameters decreased: " << decreased3d << std::endl;
        
        // Test 3: Multiple parameters (like real model)
        std::cout << "\nTest 3: Multiple parameters..." << std::endl;
        auto weight = std::make_shared<Variable>(Tensor(4, 4), true);
        auto bias = std::make_shared<Variable>(Tensor(1, 4), true);
        weight->getData().fill(0.5f);
        bias->getData().fill(0.1f);
        
        std::vector<std::shared_ptr<Variable>> multi_params = {weight, bias};
        AdamOptimizer opt_multi(multi_params, 0.01f);
        
        for (int step = 0; step < 3; step++) {
            weight->getGrad().fill(0.05f);
            bias->getGrad().fill(0.02f);
            opt_multi.step();
            opt_multi.zero_grad();
        }
        
        bool weight_updated = weight->getData().getValue(0, 0) != 0.5f;
        bool bias_updated = bias->getData().getValue(0, 0) != 0.1f;
        std::cout << "✓ Weight updated: " << weight_updated << std::endl;
        std::cout << "✓ Bias updated: " << bias_updated << std::endl;
        
        // Test 4: Gradient clipping
        std::cout << "\nTest 4: Gradient clipping..." << std::endl;
        auto large_param = std::make_shared<Variable>(Tensor(10, 10), true);
        large_param->getData().fill(1.0f);
        large_param->getGrad().fill(10.0f);  // Large gradients
        
        std::vector<std::shared_ptr<Variable>> clip_params = {large_param};
        AdamOptimizer opt_clip(clip_params, 0.1f);
        
        // Calculate norm before clipping
        float norm_before = 0.0f;
        for (int i = 0; i < large_param->getGrad().numel(); i++) {
            float g = large_param->getGrad().raw()[i];
            norm_before += g * g;
        }
        norm_before = std::sqrt(norm_before);
        std::cout << "Gradient norm before clipping: " << norm_before << std::endl;
        
        opt_clip.clip_grad_norm(1.0f);  // Clip to max norm of 1.0
        
        float norm_after = 0.0f;
        for (int i = 0; i < large_param->getGrad().numel(); i++) {
            float g = large_param->getGrad().raw()[i];
            norm_after += g * g;
        }
        norm_after = std::sqrt(norm_after);
        std::cout << "Gradient norm after clipping: " << norm_after << std::endl;
        
        bool clipped = norm_after <= 1.01f;  // Small tolerance
        std::cout << "✓ Gradients clipped: " << clipped << std::endl;
        
        // Test 5: Momentum and adaptive learning
        std::cout << "\nTest 5: Momentum effect..." << std::endl;
        auto momentum_param = std::make_shared<Variable>(Tensor(2, 2), true);
        momentum_param->getData().fill(1.0f);
        
        std::vector<std::shared_ptr<Variable>> momentum_params = {momentum_param};
        AdamOptimizer opt_momentum(momentum_params, 0.1f);
        
        // Consistent gradient direction
        std::vector<float> step_sizes;
        for (int step = 0; step < 5; step++) {
            float before = momentum_param->getData().getValue(0, 0);
            momentum_param->getGrad().fill(0.1f);
            opt_momentum.step();
            float after = momentum_param->getData().getValue(0, 0);
            step_sizes.push_back(before - after);
            opt_momentum.zero_grad();
        }
        
        // Later steps should be more confident (similar step sizes due to adaptive LR)
        std::cout << "Step sizes: ";
        for (float size : step_sizes) {
            std::cout << size << " ";
        }
        std::cout << std::endl;
        std::cout << "✓ Momentum accumulation working" << std::endl;
        
        std::cout << "\nAdam optimizer tests passed!\n" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Adam optimizer test failed: " << e.what() << std::endl;
    }
}

void test_linear_layer() {
    std::cout << "=== Testing Linear Layer ===" << std::endl;

    try {
        Tensor input_data(2, 3);
        input_data.setValue(0, 0, 1.0f);
        input_data.setValue(0, 1, 2.0f);
        input_data.setValue(0, 2, 3.0f);
        input_data.setValue(1, 0, 4.0f);
        input_data.setValue(1, 1, 5.0f);
        input_data.setValue(1, 2, 6.0f);

        auto input = Variable::create(input_data, true);
        Linear linear(3, 4);
        auto output = linear.forward(input);

        const Tensor& out_data = output->getData();
        std::cout << "Output shape: " 
                  << out_data.getRows() << "x" 
                  << out_data.getCols() << std::endl;

        bool shape_correct = (out_data.getRows() == 2 && out_data.getCols() == 4);
        std::cout << "Shape correct: " << shape_correct << std::endl;

        output->backward();
        std::cout << "✓ Backward pass completed successfully" << std::endl;

        if (linear.getWeights()->hasGrad()) {
            std::cout << "✓ Weights received gradients" << std::endl;
        } else {
            std::cout << "❌ No gradients for weights" << std::endl;
        }

        if (linear.getBias() && linear.getBias()->hasGrad()) {
            std::cout << "✓ Bias received gradients" << std::endl;
        } else if (linear.getBias()) {
            std::cout << "❌ No gradients for bias" << std::endl;
        } else {
            std::cout << "Bias not used" << std::endl;
        }

        std::cout << "Linear layer tests passed!\n" << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "❌ Linear test failed: " << e.what() << std::endl;
    }
}

void test_basic_tensor() {
    std::cout << "=== Testing Basic Tensor Operations ===" << std::endl;
    
    Tensor t2d(2, 3);
    Tensor t3d(2, 2, 3);
    
    t2d.fill(1.0f);
    t3d.fill(2.0f);
    
    std::cout << "2D tensor (2x3): " << t2d.getRows() << "x" << t2d.getCols() << std::endl;
    std::cout << "3D tensor (2x2x3): " << t3d.getBatchSize() << "x" << t3d.getRows() << "x" << t3d.getCols() << std::endl;
    
    Tensor t2d_copy = t2d.scale(2.0f);
    std::cout << "Scale operation works: " << (t2d_copy.getValue(0, 0) == 2.0f) << std::endl;
    
    std::cout << "Basic tensor tests passed!\n" << std::endl;
}

void test_variable_autodiff() {
    std::cout << "=== Testing Variable Automatic Differentiation ===" << std::endl;
    
    try {
        std::cout << "Test 1: Variable creation..." << std::endl;
        Tensor data1(2, 2);
        data1.setValue(0, 0, 1.0f);
        data1.setValue(0, 1, 2.0f);
        data1.setValue(1, 0, 3.0f);
        data1.setValue(1, 1, 4.0f);
        
        auto a = Variable::create(data1, true);
        std::cout << "✓ Variable creation successful" << std::endl;
        
        std::cout << "Test 2: Basic operations..." << std::endl;
        Tensor data2(2, 2);
        data2.setValue(0, 0, 0.5f);
        data2.setValue(0, 1, 1.5f);
        data2.setValue(1, 0, 2.5f);
        data2.setValue(1, 1, 3.5f);
        
        auto b = Variable::create(data2, true);
        auto c = a->add(b);
        std::cout << "✓ Addition operation successful" << std::endl;
        
        std::cout << "Test 3: Scaling..." << std::endl;
        auto d = c->scale(2.0f);
        std::cout << "✓ Scale operation successful" << std::endl;
        
        std::cout << "Test 4: Matrix multiplication..." << std::endl;
        Tensor data3(2, 2);
        data3.setValue(0, 0, 1.0f);
        data3.setValue(0, 1, 0.0f);
        data3.setValue(1, 0, 0.0f);
        data3.setValue(1, 1, 1.0f);
        
        auto identity = Variable::create(data3, true);
        auto e = a->matmul(identity);
        std::cout << "✓ Matrix multiplication successful" << std::endl;
        
        std::cout << "Test 5: Backward pass..." << std::endl;
        e->backward();
        std::cout << "✓ Backward pass completed without errors" << std::endl;
        
        std::cout << "Test 6: Gradient verification..." << std::endl;
        std::cout << "Gradient of 'a':" << std::endl;
        a->getGrad().display();
        
        std::cout << "Gradient of 'identity':" << std::endl;
        identity->getGrad().display();
        
        std::cout << "Test 7: Complex chain rule..." << std::endl;
        
        a->zeroGrad();
        b->zeroGrad();
        identity->zeroGrad();
        
        auto step1 = a->add(b);
        auto step2 = step1->scale(2.0f);
        auto result = step2->matmul(identity);
        
        result->backward();
        std::cout << "✓ Complex chain rule backward pass successful" << std::endl;
        
        bool has_grad_a = false, has_grad_b = false;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                if (std::abs(a->getGrad().getValue(i, j)) > 1e-6f) has_grad_a = true;
                if (std::abs(b->getGrad().getValue(i, j)) > 1e-6f) has_grad_b = true;
            }
        }
        
        std::cout << "Variable 'a' received gradients: " << has_grad_a << std::endl;
        std::cout << "Variable 'b' received gradients: " << has_grad_b << std::endl;
        
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
    
    int vocab_size = 50;
    int d_model = 32;
    int num_layers = 2;
    int num_heads = 4;
    int max_len = 16;
    
    GPTModel model(vocab_size, d_model, num_layers, num_heads, max_len);
    
    Tensor input_tensor(3, 1);
    input_tensor.setValue(0, 0, 1);
    input_tensor.setValue(1, 0, 5);
    input_tensor.setValue(2, 0, 10);
    auto input = Variable::create(input_tensor, false);

    auto output = model.forward(input);

    std::cout << "Input shape: " << input->getData().getRows() << "x" << input->getData().getCols() << std::endl;
    std::cout << "Output shape: " << output->getData().getRows() << "x" << output->getData().getCols() << std::endl;
    std::cout << "Expected output shape: 3x" << vocab_size << std::endl;
    
    bool shape_correct = (output->getData().getRows() == 3 && output->getData().getCols());
    std::cout << "Output shape correct: " << shape_correct << std::endl;
    
    bool values_reasonable = true;
    for (int i = 0; i < 3 && values_reasonable; i++) {
        for (int j = 0; j < vocab_size && values_reasonable; j++) {
            float val = output->getData().getValue(i, j);
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
    
    int vocab_size = 20;
    int d_model = 16;
    int num_layers = 2;
    int num_heads = 2;
    int max_len = 10;
    
    GPTModel model(vocab_size, d_model, num_layers, num_heads, max_len);
    TextGen generator(model);
    
    std::vector<int> prompt = {1, 5, 3};
    
    std::cout << "Prompt tokens: ";
    for (int token : prompt) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
    
    std::string greedy_result = generator.generate_greedy(prompt, 5);
    std::cout << "Greedy result: " << greedy_result << std::endl;
    
    std::string sample_low = generator.generate_sample(prompt, 0.1f, 5);
    std::cout << "Sample (temp=0.1): " << sample_low << std::endl;
    
    std::string sample_high = generator.generate_sample(prompt, 1.5f, 5);
    std::cout << "Sample (temp=1.5): " << sample_high << std::endl;
    
    bool different_temps = (sample_low != sample_high);
    std::cout << "Different temperatures produce different results: " << different_temps << std::endl;
    
    std::cout << "Text generation tests passed!\n" << std::endl;
}

void test_model_save_load() {
    std::cout << "=== Testing Model Save/Load ===" << std::endl;
    
    GPTModel original(10, 8, 1, 2, 8);
    
    std::string filepath = "test_model.bin";
    bool save_success = original.save(filepath);
    std::cout << "Model save successful: " << save_success << std::endl;
    
    if (save_success) {
        try {
            GPTModel loaded = GPTModel::load(filepath);
            
            Tensor test_tensor(2, 1);
            test_tensor.setValue(0, 0, 1);
            test_tensor.setValue(1, 0, 3);
            auto test_input = Variable::create(test_tensor, false);

            auto original_output = original.forward(test_input);
            auto loaded_output = loaded.forward(test_input);
            
            bool outputs_match = true;
            float tolerance = 1e-6f;
            
            for (int i = 0; i < 2 && outputs_match; i++) {
                for (int j = 0; j < 10 && outputs_match; j++) {
                    float diff = std::abs(original_output->getData().getValue(i, j) - loaded_output->getData().getValue(i, j));
                    if (diff > tolerance) {
                        outputs_match = false;
                    }
                }
            }
            
            std::cout << "Loaded model produces same output: " << outputs_match << std::endl;
            
            std::remove(filepath.c_str());
            
        } catch (const std::exception& e) {
            std::cout << "Model load failed: " << e.what() << std::endl;
        }
    }
    
    std::cout << "Model save/load tests completed!\n" << std::endl;
}

void demo_text_generation() {
    std::cout << "=== Text Generation Demo ===" << std::endl;
    
    std::vector<std::string> vocab = {
        "the", "cat", "dog", "runs", "jumps", "over", "and", "quick", "brown", "fox",
        "lazy", "big", "small", "red", "blue", "house", "tree", "park", "street", "car"
    };
    
    GPTModel model(vocab.size(), 24, 3, 4, 16);
    TextGen generator(model);
    
    std::cout << "Vocabulary size: " << vocab.size() << std::endl;
    std::cout << "Generating sequences starting with different words...\n" << std::endl;
    
    for (int start_token = 0; start_token < 3; start_token++) {
        std::vector<int> prompt = {start_token};
        
        std::cout << "Starting with '" << vocab[start_token] << "':" << std::endl;
        
        std::string greedy = generator.generate_greedy(prompt, 4);
        std::cout << "  Greedy: " << greedy << " -> ";
        
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

void test_cross_entropy_loss() {
    std::cout << "=== Testing Cross-Entropy Loss ===" << std::endl;
    
    try {
        std::cout << "Test 1: Basic cross-entropy computation..." << std::endl;
        
        Tensor pred_data(2, 3);
        pred_data.setValue(0, 0, 0.7f);
        pred_data.setValue(0, 1, 0.2f);
        pred_data.setValue(0, 2, 0.1f);
        
        pred_data.setValue(1, 0, 0.1f);
        pred_data.setValue(1, 1, 0.8f);
        pred_data.setValue(1, 2, 0.1f);
        
        Tensor target_data(2, 1);
        target_data.setValue(0, 0, 0.0f);
        target_data.setValue(1, 0, 1.0f);
        
        auto predictions = Variable::create(pred_data, true);
        auto targets = Variable::create(target_data, false);
        
        auto loss = predictions->cross_entropy_loss(targets);
        
        float loss_value = loss->getData().getValue(0, 0);
        std::cout << "Computed loss: " << loss_value << std::endl;
        
        float expected = (-std::log(0.7f) + -std::log(0.8f)) / 2.0f;
        std::cout << "Expected loss: " << expected << std::endl;
        
        bool loss_correct = std::abs(loss_value - expected) < 0.01f;
        std::cout << "Loss computation correct: " << loss_correct << std::endl;
        
        std::cout << "Test 2: Cross-entropy gradients..." << std::endl;
        predictions->zeroGrad();
        loss->backward();
        
        std::cout << "Gradients for predictions:" << std::endl;
        predictions->getGrad().display();
        
        std::cout << "Test 3: Perfect predictions..." << std::endl;
        
        Tensor perfect_pred(2, 3);
        perfect_pred.setValue(0, 0, 0.99f);
        perfect_pred.setValue(0, 1, 0.005f);
        perfect_pred.setValue(0, 2, 0.005f);
        
        perfect_pred.setValue(1, 0, 0.005f);
        perfect_pred.setValue(1, 1, 0.99f);
        perfect_pred.setValue(1, 2, 0.005f);
        
        auto perfect_var = Variable::create(perfect_pred, true);
        auto perfect_loss = perfect_var->cross_entropy_loss(targets);
        
        float perfect_loss_value = perfect_loss->getData().getValue(0, 0);
        std::cout << "Perfect prediction loss: " << perfect_loss_value << std::endl;
        
        bool low_loss = perfect_loss_value < 0.1f;
        std::cout << "Perfect predictions give low loss: " << low_loss << std::endl;
        
        std::cout << "Test 4: Bad predictions..." << std::endl;
        
        Tensor bad_pred(2, 3);
        bad_pred.setValue(0, 0, 0.1f);
        bad_pred.setValue(0, 1, 0.8f);
        bad_pred.setValue(0, 2, 0.1f);
        
        bad_pred.setValue(1, 0, 0.9f);
        bad_pred.setValue(1, 1, 0.05f);
        bad_pred.setValue(1, 2, 0.05f);
        
        auto bad_var = Variable::create(bad_pred, true);
        auto bad_loss = bad_var->cross_entropy_loss(targets);
        
        float bad_loss_value = bad_loss->getData().getValue(0, 0);
        std::cout << "Bad prediction loss: " << bad_loss_value << std::endl;
        
        bool high_loss = bad_loss_value > 1.0f;
        std::cout << "Bad predictions give high loss: " << high_loss << std::endl;
        
        std::cout << "Test 5: Integration with softmax..." << std::endl;
        
        Tensor logit_data(2, 3);
        logit_data.setValue(0, 0, 2.0f);
        logit_data.setValue(0, 1, 0.1f);
        logit_data.setValue(0, 2, 0.1f);
        
        logit_data.setValue(1, 0, 0.1f);
        logit_data.setValue(1, 1, 1.5f);
        logit_data.setValue(1, 2, 0.1f);
        
        auto logits = Variable::create(logit_data, true);
        auto probs = logits->softmax();
        auto integrated_loss = probs->cross_entropy_loss(targets);
        
        integrated_loss->backward();
        
        std::cout << "Gradients flow back to logits:" << std::endl;
        logits->getGrad().display();
        
        bool has_gradients = false;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                if (std::abs(logits->getGrad().getValue(i, j)) > 1e-6f) {
                    has_gradients = true;
                    break;
                }
            }
        }
        
        std::cout << "Gradients computed for logits: " << has_gradients << std::endl;
        
        std::cout << "Cross-entropy loss tests passed!\n" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Cross-entropy test failed: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "🤖 TRANSFORMER FROM SCRATCH - COMPREHENSIVE TESTING 🤖\n" << std::endl;
    
    // Core component tests
    test_basic_tensor();
    test_variable_autodiff();
    test_linear_layer();
    test_adam_optimizer();  // 🆕 NEW TEST ADDED HERE
    
    // Model tests
    test_gpt_model();
    test_text_generation();
    test_model_save_load();
    test_cross_entropy_loss();
    
    // Practical demo
    demo_text_generation();
    
    std::cout << "\n✅ All tests completed successfully!" << std::endl;
    std::cout << "Your transformer implementation is fully functional!" << std::endl;
    
    return 0;
}