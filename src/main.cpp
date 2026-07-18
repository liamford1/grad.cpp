#include "transformer/gpt_model.h"
#include "transformer/text_gen.h"
#include "tokenizer/bpe_tokenizer.h"
#include "data/dataset.h"
#include "data/dataloader.h"
#include "training/trainer.h"
#include "utils/metrics.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

std::string read_text_file(const std::string& data_path) {
    std::cout << "Reading " << data_path << "..." << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
    std::ifstream file(data_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open " + data_path);
    }
    std::string text((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
    file.close();
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << " " << (text.size() / 1024) << "KB (" << ms << "ms)" << std::endl;
    return text;
}

void load_tokenizer(const std::string& text,
                    const std::string& cache_prefix,
                    int vocab_size,
                    BPETokenizer& tokenizer) {
    std::string cache_file = cache_prefix + "_" + std::to_string(vocab_size) + ".cache";
    std::ifstream cache_check(cache_file);

    if (cache_check.good()) {
        cache_check.close();
        std::cout << "Loading tokenizer from cache..." << std::flush;
        auto start = std::chrono::high_resolution_clock::now();
        tokenizer.load(cache_file);
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << " done (" << ms << "ms)" << std::endl;
    } else {
        std::cout << "Training new tokenizer..." << std::flush;
        auto start = std::chrono::high_resolution_clock::now();
        tokenizer.train(text);
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << " done (" << ms << "ms)" << std::endl;
        tokenizer.save(cache_file);
        std::cout << "Cached to " << cache_file << std::endl;
    }

    std::cout << "Vocab size: " << tokenizer.getCurrentVocabSize() << std::endl;
}

void load_data_and_tokenizer(const std::string& data_path,
                              const std::string& cache_prefix,
                              int vocab_size,
                              std::string& text,
                              BPETokenizer& tokenizer,
                              std::vector<int>& tokens) {
    utils::print_section("Loading Data");

    text = read_text_file(data_path);
    load_tokenizer(text, cache_prefix, vocab_size, tokenizer);

    std::cout << "Encoding text..." << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
    tokens = tokenizer.encode(text);
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
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

int run_generation(const std::string& checkpoint_path, const std::string& prompt) {
    std::cout << "\nTransformer Generation\n" << std::endl;

    try {
        const int vocab_size = 5000;

        BPETokenizer tokenizer(vocab_size);
        std::string text = read_text_file("data/shakespeare.txt");
        load_tokenizer(text, "tokenizer", vocab_size, tokenizer);

        utils::print_section("Loading Model");
        GPTModel model = GPTModel::load(checkpoint_path);

        TextGen generator(model, &tokenizer);
        auto prompt_tokens = tokenizer.encode(prompt);

        std::cout << "\n--- Greedy Decoding ---\n" << std::endl;
        std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
        std::cout << generator.generate_greedy(prompt_tokens, 150) << std::endl;

        std::cout << "\n--- Sampling (temp=0.8) ---\n" << std::endl;
        std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
        std::cout << generator.generate_sample(prompt_tokens, 0.8f, 150) << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}

int run_training(bool fast_mode) {
    std::cout << "\nTransformer Training\n" << std::endl;

    try {
        const int vocab_size = fast_mode ? 500 : 5000;
        const int num_steps = fast_mode ? 50 : 50000;
        const int seq_length = fast_mode ? 64 : 96;

        std::string text;
        BPETokenizer tokenizer(vocab_size);
        std::vector<int> tokens;

        load_data_and_tokenizer("data/shakespeare.txt", "tokenizer",
                                vocab_size, text, tokenizer, tokens);

        utils::print_section("Initializing Model");

        training::TrainingConfig config;
        config.vocab_size = tokenizer.getCurrentVocabSize();
        config.d_model = fast_mode ? 128 : 512;
        config.num_layers = fast_mode ? 2 : 6;
        config.num_heads = fast_mode ? 4 : 8;
        config.max_len = 1024;
        config.seq_length = seq_length;
        config.batch_size = fast_mode ? 4 : 8;
        config.learning_rate = 3e-4f;
        config.dropout = 0.1f;
        config.warmup_steps = fast_mode ? 10 : 500;
        config.num_steps = num_steps;
        config.checkpoint_interval = 2500;
        config.checkpoint_prefix = fast_mode ? "shakespeare_fast" : "shakespeare";

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

int main(int argc, char* argv[]) {
    std::string mode = (argc > 1) ? argv[1] : "";

    if (mode == "train") {
        return run_training(false);
    }
    if (mode == "train-fast") {
        return run_training(true);
    }
    if (mode == "generate") {
        std::string checkpoint = (argc > 2) ? argv[2] : "shakespeare_final.bin";
        std::string prompt = (argc > 3) ? argv[3] : "ROMEO:\n";
        return run_generation(checkpoint, prompt);
    }

    std::cerr << "Usage: " << argv[0] << " <mode>\n"
              << "  train                            full training run on data/shakespeare.txt\n"
              << "  train-fast                       small config for a quick smoke test\n"
              << "  generate [checkpoint] [prompt]   sample from a saved checkpoint\n";
    return 1;
}
