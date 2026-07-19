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

// Inference-time tokenizer loading: the cache must exist (train/prepare
// created it), so the corpus text - possibly gigabytes - is never read.
// Falls back to training from text only for small unprepared corpora.
void load_tokenizer_for_inference(const std::string& corpus_path, int vocab_size,
                                  BPETokenizer& tokenizer) {
    std::string cache_file = tokenizer_cache_prefix(corpus_path) + "_"
                           + std::to_string(vocab_size) + ".cache";
    std::ifstream cache_check(cache_file);
    if (cache_check.good()) {
        cache_check.close();
        tokenizer.load(cache_file);
        std::cout << "Tokenizer: " << cache_file << " (vocab "
                  << tokenizer.getCurrentVocabSize() << ")" << std::endl;
        return;
    }
    std::string text = read_text_file(corpus_path);
    load_tokenizer(text, tokenizer_cache_prefix(corpus_path), vocab_size, tokenizer);
}

int run_generation(const std::string& checkpoint_path, const std::string& prompt,
                   const std::string& corpus_path, int vocab_size) {
    std::cout << "\nTransformer Generation\n" << std::endl;

    try {
        BPETokenizer tokenizer(vocab_size);
        load_tokenizer_for_inference(corpus_path, vocab_size, tokenizer);

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
int run_chat(const std::string& checkpoint_path,
             const std::string& corpus_path, int vocab_size) {
    std::cout << "\nTransformer Chat\n" << std::endl;

    try {
        BPETokenizer tokenizer(vocab_size);
        load_tokenizer_for_inference(corpus_path, vocab_size, tokenizer);

        utils::print_section("Loading Model");
        GPTModel model = GPTModel::load(checkpoint_path);
        TextGen generator(model, &tokenizer);

        std::cout << "\nThis is a base language model: it continues text in the"
                  << " style of its training corpus.\nEmpty line or 'exit' quits.\n" << std::endl;

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

// Pre-tokenize a corpus once: train (or load) the BPE tokenizer, encode the
// whole text, and write 95/5 train/val token files. Training then memory-
// maps those files instead of re-encoding the corpus on every run.
int run_prepare(const std::string& corpus_path, int vocab_size) {
    std::cout << "\nTransformer Prepare\n" << std::endl;

    try {
        utils::print_section("Tokenizing corpus");
        std::string text = read_text_file(corpus_path);
        BPETokenizer tokenizer(vocab_size);

        // BPE merge learning scans every unique word once per merge, so its
        // cost grows with corpus size for no statistical benefit: token
        // frequencies converge long before 32MB. Train on a prefix sample
        // (cut at a word boundary), then encode the full corpus with it.
        constexpr size_t kTokenizerSampleBytes = 32ull * 1024 * 1024;
        if (text.size() > kTokenizerSampleBytes) {
            size_t cut = text.rfind(' ', kTokenizerSampleBytes);
            if (cut == std::string::npos) cut = kTokenizerSampleBytes;
            std::cout << "Corpus is " << (text.size() >> 20) << "MB; training tokenizer on a "
                      << (cut >> 20) << "MB sample" << std::endl;
            std::string sample = text.substr(0, cut);
            load_tokenizer(sample, tokenizer_cache_prefix(corpus_path), vocab_size, tokenizer);
        } else {
            load_tokenizer(text, tokenizer_cache_prefix(corpus_path), vocab_size, tokenizer);
        }

        std::cout << "Encoding text..." << std::flush;
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<int> tokens = tokenizer.encode(text);
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << " " << tokens.size() << " tokens (" << ms << "ms)" << std::endl;

        // Release the raw text before writing; the token ids are all that
        // remain live, and the split writes directly from ranges of them.
        text.clear();
        text.shrink_to_fit();

        size_t split = tokens.size() * 95 / 100;
        int vocab = tokenizer.getCurrentVocabSize();
        std::string train_bin = token_bin_path(corpus_path, vocab_size, "train");
        std::string val_bin = token_bin_path(corpus_path, vocab_size, "val");
        tokenfile::write(train_bin, tokens.data(), split, vocab);
        tokenfile::write(val_bin, tokens.data() + split, tokens.size() - split, vocab);

        std::cout << "\nWrote " << train_bin << " (" << split << " tokens)"
                  << "\nWrote " << val_bin << " (" << (tokens.size() - split) << " tokens)"
                  << "\n\nTrain with: ./build/transformer train " << corpus_path
                  << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}

// Model/training presets. "small" is the original 22M-param Shakespeare
// config; "medium" (~70M params) is sized so its FFN and logits matmuls
// cross the ~10 GFLOP threshold where the Metal GPU backend starts winning
// (BENCHMARKS.md #7). "fast" is the CI smoke test.
struct Preset {
    const char* name;
    int vocab_size;
    int d_model, num_layers, num_heads;
    int max_len, seq_length, batch_size;
    float learning_rate;
    int warmup_steps, num_steps;
    int checkpoint_interval, eval_interval;
    int max_eval_batches;  // 0 = evaluate the whole val set
};

// small's step count is set by measurement: on the 254K-token Shakespeare
// corpus, val perplexity bottoms around step 6000 and rises after (a 22M
// model memorizes a corpus that small). 8000 steps lets the cosine
// schedule finish near the minimum instead of training 8x past it.
const Preset kPresets[] = {
    {"fast",   500,   128, 2,  4,  1024, 64,  4,  3e-4f, 10,   50,    2500, 25,  0},
    {"small",  5000,  512, 6,  8,  1024, 96,  8,  3e-4f, 500,  8000,  2500, 250, 0},
    {"medium", 16000, 768, 8,  12, 1024, 256, 16, 3e-4f, 1000, 20000, 2000, 250, 32},
};

const Preset* find_preset(const std::string& name) {
    for (const auto& p : kPresets) {
        if (name == p.name) return &p;
    }
    return nullptr;
}

// Checkpoint prefix from the corpus filename: data/tinystories.txt ->
// "tinystories" (data/shakespeare.txt keeps its historical "shakespeare").
std::string checkpoint_stem(const std::string& corpus_path) {
    size_t slash = corpus_path.find_last_of('/');
    std::string base = (slash == std::string::npos) ? corpus_path
                                                    : corpus_path.substr(slash + 1);
    size_t dot = base.find_last_of('.');
    return (dot == std::string::npos) ? base : base.substr(0, dot);
}

int run_training(const Preset& preset, const std::string& corpus_path) {
    std::cout << "\nTransformer Training (" << preset.name << ")\n" << std::endl;

    try {
        const int vocab_size = preset.vocab_size;
        const int num_steps = preset.num_steps;
        const int seq_length = preset.seq_length;
        const bool fast_mode = std::string(preset.name) == "fast";

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
        config.d_model = preset.d_model;
        config.num_layers = preset.num_layers;
        config.num_heads = preset.num_heads;
        config.max_len = preset.max_len;
        config.seq_length = seq_length;
        config.batch_size = preset.batch_size;
        config.learning_rate = preset.learning_rate;
        config.dropout = 0.1f;
        config.warmup_steps = preset.warmup_steps;
        config.num_steps = num_steps;
        config.checkpoint_interval = preset.checkpoint_interval;
        config.checkpoint_prefix = checkpoint_stem(corpus_path)
                                 + (fast_mode ? "_fast" : "");
        config.eval_interval = preset.eval_interval;
        config.max_eval_batches = preset.max_eval_batches;

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
        std::string corpus = (argc > 2) ? argv[2] : default_corpus;
        std::string preset_name = (argc > 3) ? argv[3] : "small";
        const Preset* preset = find_preset(preset_name);
        if (!preset || preset_name == "fast") {
            std::cerr << "Unknown preset '" << preset_name << "' (available: small, medium)" << std::endl;
            return 1;
        }
        return run_training(*preset, corpus);
    }
    if (mode == "train-fast") {
        return run_training(*find_preset("fast"), (argc > 2) ? argv[2] : default_corpus);
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
        std::string corpus = (argc > 4) ? argv[4] : default_corpus;
        int vocab = (argc > 5) ? std::atoi(argv[5]) : 5000;
        return run_generation(checkpoint, prompt, corpus, vocab);
    }
    if (mode == "bench") {
        int steps = (argc > 2) ? std::atoi(argv[2]) : 20;
        return run_benchmark(steps);
    }
    if (mode == "chat") {
        std::string checkpoint = (argc > 2) ? argv[2] : "shakespeare_final.bin";
        std::string corpus = (argc > 3) ? argv[3] : default_corpus;
        int vocab = (argc > 4) ? std::atoi(argv[4]) : 5000;
        return run_chat(checkpoint, corpus, vocab);
    }

    std::cerr << "Usage: " << argv[0] << " <mode>\n"
              << "  prepare <corpus.txt> [vocab]         pre-tokenize a corpus to .bin token files\n"
              << "  train [corpus.txt] [small|medium]    full training run (uses .bin files if present)\n"
              << "  train-fast [corpus.txt]              tiny config for a quick smoke test\n"
              << "  generate [ckpt] [prompt] [corpus] [vocab]   sample from a saved checkpoint\n"
              << "  chat [ckpt] [corpus] [vocab]                interactive prompt/continue REPL\n"
              << "  bench [steps]                        measure training and generation speed\n";
    return 1;
}
