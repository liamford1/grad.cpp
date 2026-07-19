#include "transformer/gpt_model.h"
#include "transformer/text_gen.h"
#include "tokenizer/bpe_tokenizer.h"
#include "data/dataset.h"
#include "data/dataloader.h"
#include "data/token_file.h"
#include "training/trainer.h"
#include "utils/metrics.h"
#include "utils/training_utils.h"
#include <cstdlib>
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

// Interactive REPL: type a prompt, watch the model continue it token by
// token. Note this is a base language model, not an instruction-tuned
// assistant: it continues text in the style of its training corpus rather
// than answering questions.
int run_chat(const std::string& checkpoint_path) {
    std::cout << "\nTransformer Chat\n" << std::endl;

    try {
        const int vocab_size = 5000;

        BPETokenizer tokenizer(vocab_size);
        std::string text = read_text_file("data/shakespeare.txt");
        load_tokenizer(text, "tokenizer", vocab_size, tokenizer);

        utils::print_section("Loading Model");
        GPTModel model = GPTModel::load(checkpoint_path);
        TextGen generator(model, &tokenizer);

        std::cout << "\nThis model continues text in the style of its training data"
                  << " (Shakespeare).\nTry a prompt like \"ROMEO:\" or"
                  << " \"First Citizen:\". Empty line or 'exit' quits.\n" << std::endl;

        std::string line;
        while (true) {
            std::cout << "> " << std::flush;
            if (!std::getline(std::cin, line)) break;
            if (line.empty() || line == "exit" || line == "quit") break;

            auto prompt = tokenizer.encode(line + "\n");
            std::cout << line << std::flush;
            generator.generate_stream(
                prompt,
                [](const std::string& piece) { std::cout << piece << std::flush; },
                0.8f, 200);
            std::cout << "\n" << std::endl;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}

// Repeatable performance benchmark: times raw training steps and token
// generation at the full model config, without writing checkpoints.
int run_benchmark(int bench_steps) {
    std::cout << "\nTransformer Benchmark\n" << std::endl;

    try {
        const int vocab_size = 5000;
        const int warmup_steps = 3;

        BPETokenizer tokenizer(vocab_size);
        std::string text = read_text_file("data/shakespeare.txt");
        load_tokenizer(text, "tokenizer", vocab_size, tokenizer);
        std::vector<int> tokens = tokenizer.encode(text);

        const int d_model = 512, num_layers = 6, num_heads = 8;
        const int max_len = 1024, seq_length = 96, batch_size = 8;

        GPTModel model(vocab_size, d_model, num_layers, num_heads, max_len, 0.1f);
        auto params = model.getAllParameters();
        AdamOptimizer optimizer(params, 3e-4f, 0.9f, 0.999f, 1e-8f, 0.0f);

        auto dataset = std::make_shared<TextDataset>(tokens, seq_length);
        DataLoader loader(dataset, batch_size, true);

        utils::print_section("Training throughput");
        std::cout << "Config: d_model=" << d_model << " layers=" << num_layers
                  << " heads=" << num_heads << " seq=" << seq_length
                  << " batch=" << batch_size << std::endl;

        auto run_step = [&]() {
            if (!loader.has_next()) loader.reset();
            auto batch = loader.next_batch();
            auto in = Variable::create(batch.input, false);
            auto tgt = Variable::create(batch.target, false);
            auto logits = model.forward(in, true);
            auto loss = logits->log_softmax()->nll_loss(tgt);
            optimizer.zero_grad();
            loss->backward();
            loss->release_graph();
            optimizer.clip_grad_norm(5.0f);
            optimizer.step();
        };

        for (int i = 0; i < warmup_steps; i++) run_step();

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < bench_steps; i++) run_step();
        auto end = std::chrono::high_resolution_clock::now();
        double train_s = std::chrono::duration<double>(end - start).count();

        double steps_per_s = bench_steps / train_s;
        double tokens_per_s = steps_per_s * batch_size * seq_length;
        std::cout << bench_steps << " steps in " << train_s << "s" << std::endl;
        std::cout << "  " << steps_per_s << " steps/s" << std::endl;
        std::cout << "  " << tokens_per_s << " tokens/s" << std::endl;

        utils::print_section("Generation throughput");
        const int gen_tokens = 64;
        TextGen generator(model, &tokenizer);
        auto prompt = tokenizer.encode("ROMEO:\n");

        start = std::chrono::high_resolution_clock::now();
        generator.generate_sample(prompt, 0.8f, gen_tokens);
        end = std::chrono::high_resolution_clock::now();
        double gen_s = std::chrono::duration<double>(end - start).count();
        std::cout << gen_tokens << " tokens in " << gen_s << "s ("
                  << (gen_tokens / gen_s) << " tok/s)" << std::endl;

        std::cout << "\nPeak memory: " << utils::get_memory_mb() << " MB" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}

// The default corpus keeps its historical cache name so existing caches
// and checkpoints stay valid; other corpora get corpus-derived names.
std::string tokenizer_cache_prefix(const std::string& corpus_path) {
    if (corpus_path == "data/shakespeare.txt") return "tokenizer";
    return corpus_path + ".tokenizer";
}

std::string token_bin_path(const std::string& corpus_path, int vocab_size,
                           const std::string& split) {
    return corpus_path + "." + std::to_string(vocab_size) + "." + split + ".bin";
}

// Pre-tokenize a corpus once: train (or load) the BPE tokenizer, encode the
// whole text, and write 95/5 train/val token files. Training then memory-
// maps those files instead of re-encoding the corpus on every run.
int run_prepare(const std::string& corpus_path, int vocab_size) {
    std::cout << "\nTransformer Prepare\n" << std::endl;

    try {
        utils::print_section("Tokenizing corpus");
        std::string text = read_text_file(corpus_path);
        BPETokenizer tokenizer(vocab_size);
        load_tokenizer(text, tokenizer_cache_prefix(corpus_path), vocab_size, tokenizer);

        std::cout << "Encoding text..." << std::flush;
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<int> tokens = tokenizer.encode(text);
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << " " << tokens.size() << " tokens (" << ms << "ms)" << std::endl;

        size_t split = tokens.size() * 95 / 100;
        std::vector<int> train_tokens(tokens.begin(), tokens.begin() + split);
        std::vector<int> val_tokens(tokens.begin() + split, tokens.end());

        int vocab = tokenizer.getCurrentVocabSize();
        std::string train_bin = token_bin_path(corpus_path, vocab_size, "train");
        std::string val_bin = token_bin_path(corpus_path, vocab_size, "val");
        tokenfile::write(train_bin, train_tokens, vocab);
        tokenfile::write(val_bin, val_tokens, vocab);

        std::cout << "\nWrote " << train_bin << " (" << train_tokens.size() << " tokens)"
                  << "\nWrote " << val_bin << " (" << val_tokens.size() << " tokens)"
                  << "\n\nTrain with: ./build/transformer train " << corpus_path
                  << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}

int run_training(bool fast_mode, const std::string& corpus_path) {
    std::cout << "\nTransformer Training\n" << std::endl;

    try {
        const int vocab_size = fast_mode ? 500 : 5000;
        const int num_steps = fast_mode ? 50 : 50000;
        const int seq_length = fast_mode ? 64 : 96;

        BPETokenizer tokenizer(vocab_size);
        std::shared_ptr<Dataset> dataset;
        std::shared_ptr<Dataset> val_dataset;

        std::string train_bin = token_bin_path(corpus_path, vocab_size, "train");
        std::string val_bin = token_bin_path(corpus_path, vocab_size, "val");
        std::string cache_file = tokenizer_cache_prefix(corpus_path) + "_"
                               + std::to_string(vocab_size) + ".cache";

        if (tokenfile::exists(train_bin) && tokenfile::exists(val_bin)) {
            // Pre-tokenized path: memory-map the token files. The corpus
            // text is never loaded and nothing is re-encoded, so startup
            // cost and memory use are independent of corpus size.
            utils::print_section("Loading Data (pre-tokenized)");
            std::ifstream cache_check(cache_file);
            if (!cache_check.good()) {
                throw std::runtime_error("Found token files but no tokenizer cache ("
                                         + cache_file + "); run: ./build/transformer prepare "
                                         + corpus_path);
            }
            tokenizer.load(cache_file);

            auto mapped_train = std::make_shared<MappedTokenDataset>(train_bin, seq_length);
            auto mapped_val = std::make_shared<MappedTokenDataset>(val_bin, seq_length,
                                                                   seq_length);
            if (mapped_train->vocabSize() != tokenizer.getCurrentVocabSize()) {
                throw std::runtime_error("Token file vocab does not match tokenizer cache; re-run prepare");
            }
            std::cout << "Mapped " << mapped_train->tokenCount() << " train / "
                      << mapped_val->tokenCount() << " val tokens from "
                      << train_bin << std::endl;
            dataset = mapped_train;
            val_dataset = mapped_val;
        } else {
            // In-memory path: read and encode the corpus now. Fine for
            // small corpora; for anything large, run `prepare` first.
            std::string text;
            std::vector<int> tokens;
            load_data_and_tokenizer(corpus_path, tokenizer_cache_prefix(corpus_path),
                                    vocab_size, text, tokenizer, tokens);

            // Hold out the last 5% of the corpus for validation. The split
            // is contiguous, so no training window ever overlaps validation
            // text - val perplexity measures generalization, not
            // memorization.
            size_t split = tokens.size() * 95 / 100;
            std::vector<int> train_tokens(tokens.begin(), tokens.begin() + split);
            std::vector<int> val_tokens(tokens.begin() + split, tokens.end());

            dataset = std::make_shared<TextDataset>(train_tokens, seq_length);
            // Non-overlapping windows: evaluation covers the whole held-out
            // slice once, deterministically.
            val_dataset = std::make_shared<TextDataset>(val_tokens, seq_length,
                                                        seq_length);
        }

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
        config.eval_interval = fast_mode ? 25 : 250;

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

        DataLoader loader(dataset, config.batch_size, true);
        DataLoader val_loader(val_dataset, config.batch_size, false);

        std::cout << "Dataset: " << dataset->size() << " train / "
                  << val_dataset->size() << " val sequences\n" << std::endl;

        training::Trainer trainer(config, model, loader, tokenizer, &val_loader);
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

    std::string default_corpus = "data/shakespeare.txt";

    if (mode == "train") {
        return run_training(false, (argc > 2) ? argv[2] : default_corpus);
    }
    if (mode == "train-fast") {
        return run_training(true, (argc > 2) ? argv[2] : default_corpus);
    }
    if (mode == "prepare") {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " prepare <corpus.txt> [vocab_size]" << std::endl;
            return 1;
        }
        int vocab = (argc > 3) ? std::atoi(argv[3]) : 5000;
        return run_prepare(argv[2], vocab);
    }
    if (mode == "generate") {
        std::string checkpoint = (argc > 2) ? argv[2] : "shakespeare_final.bin";
        std::string prompt = (argc > 3) ? argv[3] : "ROMEO:\n";
        return run_generation(checkpoint, prompt);
    }
    if (mode == "bench") {
        int steps = (argc > 2) ? std::atoi(argv[2]) : 20;
        return run_benchmark(steps);
    }
    if (mode == "chat") {
        std::string checkpoint = (argc > 2) ? argv[2] : "shakespeare_final.bin";
        return run_chat(checkpoint);
    }

    std::cerr << "Usage: " << argv[0] << " <mode>\n"
              << "  prepare <corpus.txt> [vocab]     pre-tokenize a corpus to .bin token files\n"
              << "  train [corpus.txt]               full training run (uses .bin files if present)\n"
              << "  train-fast [corpus.txt]          small config for a quick smoke test\n"
              << "  generate [checkpoint] [prompt]   sample from a saved checkpoint\n"
              << "  chat [checkpoint]                interactive prompt/continue REPL\n"
              << "  bench [steps]                    measure training and generation speed\n";
    return 1;
}
