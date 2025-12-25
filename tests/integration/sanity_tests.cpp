#include "transformer/gpt_model.h"
#include "transformer/variable.h"
#include "transformer/optimizer.h"
#include "data/dataset.h"
#include "data/dataloader.h"
#include "tokenizer/bpe_tokenizer.h"
#include "utils/training_utils.h"
#include "utils/metrics.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>

void test_overfit_tiny_sequence() {
    utils::print_header("Overfitting Test: Memorize 5 Tokens");

    const int vocab_size = 20;
    const int d_model = 32;
    const int num_layers = 2;
    const int num_heads = 4;
    const int max_len = 10;
    const int seq_length = 5;

    GPTModel model(vocab_size, d_model, num_layers, num_heads, max_len);
    auto params = model.getAllParameters();
    AdamOptimizer optimizer(params, 0.001f, 0.9f, 0.999f, 1e-8f, 0.0f);

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
        float grad_norm = utils::compute_grad_norm(params);
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
            std::cout << "Gradients are zero!" << std::endl;
            return;
        }
    }

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
    std::cout << (acc >= 80.0f ? "SUCCESS" : "FAILED") << std::endl;
}

void test_dataloader() {
    utils::print_header("DataLoader Test: Multi-Batch Training");

    std::vector<int> tokens;
    for (int i = 0; i < 200; i++) tokens.push_back(i % 15);

    GPTModel model(15, 24, 2, 4, 16);
    auto dataset = std::make_shared<TextDataset>(tokens, 8);
    DataLoader loader(dataset, 2, true);

    auto params = model.getAllParameters();
    AdamOptimizer optimizer(params, 1e-5f, 0.9f, 0.999f, 1e-8f, 0.0f);

    for (int epoch = 0; epoch < 3; epoch++) {
        std::cout << "\nEpoch " << (epoch + 1) << std::endl;
        loader.reset();
        float total_loss = 0.0f;
        int batch_count = 0;

        while (loader.has_next()) {
            auto batch = loader.next_batch();

            int batch_size = batch.input.getBatchSize();
            int seq_len = batch.input.getRows();

            Tensor input_2d(batch_size * seq_len, 1);
            Tensor target_2d(batch_size * seq_len, 1);
            utils::reshape_batch_to_2d(batch.input, batch.target, input_2d, target_2d);

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

    std::cout << "DataLoader test complete" << std::endl;
}

void benchmark_training_speed() {
    utils::print_header("Performance Benchmark: 100 Steps");

    std::ifstream file("data/shakespeare.txt");
    if (!file.is_open()) {
        std::cerr << "Can't open data/shakespeare.txt" << std::endl;
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

    const int vocab_size = tokenizer.getCurrentVocabSize();
    const int d_model = 512;
    const int num_layers = 6;
    const int num_heads = 8;
    const int max_len = 1024;
    const int seq_length = 128;
    const int batch_size = 8;

    GPTModel model(vocab_size, d_model, num_layers, num_heads, max_len, 0.1f);
    auto params = model.getAllParameters();
    auto dataset = std::make_shared<TextDataset>(tokens, seq_length);
    DataLoader loader(dataset, batch_size, true);
    AdamOptimizer optimizer(params, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f);

    std::cout << "Config: vocab=" << vocab_size << " d_model=" << d_model
              << " layers=" << num_layers << " heads=" << num_heads << std::endl;
    std::cout << "Running 100 steps...\n" << std::endl;

    for (int step = 0; step < 5; step++) {
        if (!loader.has_next()) loader.reset();
        auto batch = loader.next_batch();

        int bs = batch.input.getBatchSize();
        int sl = batch.input.getRows();
        Tensor input_2d(bs * sl, 1), target_2d(bs * sl, 1);
        utils::reshape_batch_to_2d(batch.input, batch.target, input_2d, target_2d);

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

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < 100; step++) {
        if (!loader.has_next()) loader.reset();
        auto batch = loader.next_batch();

        int bs = batch.input.getBatchSize();
        int sl = batch.input.getRows();
        Tensor input_2d(bs * sl, 1), target_2d(bs * sl, 1);
        utils::reshape_batch_to_2d(batch.input, batch.target, input_2d, target_2d);

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

int main(int argc, char* argv[]) {
    std::cout << "Running Sanity Tests\n" << std::endl;

    try {
        if (argc > 1) {
            std::string test_name(argv[1]);
            if (test_name == "overfit") {
                test_overfit_tiny_sequence();
            } else if (test_name == "dataloader") {
                test_dataloader();
            } else if (test_name == "benchmark") {
                benchmark_training_speed();
            } else {
                std::cerr << "Unknown test: " << test_name << std::endl;
                std::cerr << "Available tests: overfit, dataloader, benchmark" << std::endl;
                return 1;
            }
        } else {
            test_overfit_tiny_sequence();
            test_dataloader();
            benchmark_training_speed();
        }

        std::cout << "\nALL TESTS COMPLETED!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}
