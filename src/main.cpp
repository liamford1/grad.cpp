#include "transformer/gpt_model.h"
#include "transformer/text_gen.h"
#include "tokenizer/bpe_tokenizer.h"
#include "data/dataset.h"
#include "data/dataloader.h"
#include "training/trainer.h"
#include "utils/metrics.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

void load_data_and_tokenizer(const std::string& data_path,
                              const std::string& cache_prefix,
                              int vocab_size,
                              std::string& text,
                              BPETokenizer& tokenizer,
                              std::vector<int>& tokens) {
    utils::print_section("Loading Data");

    std::cout << "Reading " << data_path << "..." << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
    std::ifstream file(data_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open " + data_path);
    }
    text = std::string((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    file.close();
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << " " << (text.size() / 1024) << "KB (" << ms << "ms)" << std::endl;

    std::string cache_file = cache_prefix + "_" + std::to_string(vocab_size) + ".cache";
    std::ifstream cache_check(cache_file);

    if (cache_check.good()) {
        cache_check.close();
        std::cout << "Loading tokenizer from cache..." << std::flush;
        start = std::chrono::high_resolution_clock::now();
        tokenizer.load(cache_file);
        end = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << " done (" << ms << "ms)" << std::endl;
    } else {
        std::cout << "Training new tokenizer..." << std::flush;
        start = std::chrono::high_resolution_clock::now();
        tokenizer.train(text);
        end = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << " done (" << ms << "ms)" << std::endl;
        tokenizer.save(cache_file);
        std::cout << "Cached to " << cache_file << std::endl;
    }

    std::cout << "Vocab size: " << tokenizer.getCurrentVocabSize() << std::endl;

    std::cout << "Encoding text..." << std::flush;
    start = std::chrono::high_resolution_clock::now();
    tokens = tokenizer.encode(text);
    end = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << " " << tokens.size() << " tokens (" << ms << "ms)" << std::endl;
}

void generate_samples(GPTModel& model, BPETokenizer& tokenizer) {
    utils::print_section("Generating Samples");

    TextGen generator(model, &tokenizer);

    std::vector<std::string> prompts = {
        "ROMEO:\n",
        "JULIET:\n",
        "First Citizen:\n"
    };

    std::cout << "\n--- Greedy Decoding ---\n" << std::endl;
    for (const auto& prompt_str : prompts) {
        std::cout << "Prompt: \"" << prompt_str << "\"" << std::endl;
        auto prompt = tokenizer.encode(prompt_str);
        std::string generated = generator.generate_greedy(prompt, 150);
        std::cout << generated << std::endl;
        std::cout << std::string(40, '-') << "\n" << std::endl;
    }

    std::cout << "\n--- Sampling (temp=0.8) ---\n" << std::endl;
    for (const auto& prompt_str : prompts) {
        std::cout << "Prompt: \"" << prompt_str << "\"" << std::endl;
        auto prompt = tokenizer.encode(prompt_str);
        std::string generated = generator.generate_sample(prompt, 0.8f, 150);
        std::cout << generated << std::endl;
        std::cout << std::string(40, '-') << "\n" << std::endl;
    }
}

int main() {
    std::cout << "\nTransformer Training\n" << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;
    std::cout << "VRAM: " << (prop.totalGlobalMem / (1024*1024*1024)) << " GB\n" << std::endl;

    try {
        const bool FAST_MODE = false;
        const int vocab_size = FAST_MODE ? 500 : 5000;
        const int num_steps = FAST_MODE ? 50 : 50000;
        const int seq_length = FAST_MODE ? 64 : 96;

        std::string text;
        BPETokenizer tokenizer(vocab_size);
        std::vector<int> tokens;

        load_data_and_tokenizer("data/shakespeare.txt", "tokenizer",
                                vocab_size, text, tokenizer, tokens);

        utils::print_section("Initializing Model");

        training::TrainingConfig config;
        config.vocab_size = tokenizer.getCurrentVocabSize();
        config.d_model = 512;
        config.num_layers = 6;
        config.num_heads = 8;
        config.max_len = 1024;
        config.seq_length = seq_length;
        config.batch_size = 8;
        config.learning_rate = 3e-4f;
        config.dropout = 0.0f;
        config.warmup_steps = 500;
        config.num_steps = num_steps;
        config.checkpoint_interval = 2500;
        config.checkpoint_prefix = "shakespeare";

        auto start = std::chrono::high_resolution_clock::now();
        GPTModel model(config.vocab_size, config.d_model, config.num_layers,
                       config.num_heads, config.max_len, config.dropout);
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        auto params = model.getAllParameters();
        int total_params = 0;
        for (const auto& p : params) total_params += p->getData().numel();

        std::cout << "Model initialized (" << ms << "ms)" << std::endl;
        std::cout << "Parameters: " << (total_params / 1e6f) << "M" << std::endl;

        auto dataset = std::make_shared<TextDataset>(tokens, config.seq_length);
        DataLoader loader(dataset, config.batch_size, true);
        std::cout << "Dataset: " << dataset->size() << " sequences\n" << std::endl;

        training::Trainer trainer(config, model, loader, tokenizer);
        trainer.train();

        generate_samples(model, tokenizer);

        std::cout << "\nTraining Complete!\n" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}
