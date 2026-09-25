#include "transformer/gpt_model.h"
#include "transformer/metal_backend.h"
#include "transformer/text_gen.h"
#include "tokenizer/bpe_tokenizer.h"
#include "data/dataset.h"
#include "data/dataloader.h"
#include "data/token_file.h"
#include "training/trainer.h"
#include "utils/dashboard.h"
#include "utils/metrics.h"
#include "utils/training_utils.h"
#include <algorithm>
#include <charconv>
#include <cstring>
#include <ctime>
#include <functional>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <string>
#include <chrono>
#include <sys/stat.h>

#ifndef GRAD_VERSION
#define GRAD_VERSION "dev"
#endif
#ifndef GRAD_GIT_SHA
#define GRAD_GIT_SHA "unknown"
#endif
#ifndef GRAD_BUILD_TYPE
#define GRAD_BUILD_TYPE "unknown"
#endif
#ifndef GRAD_COMPILER
#define GRAD_COMPILER "unknown"
#endif
#ifndef GRAD_SYSTEM
#define GRAD_SYSTEM "unknown"
#endif

std::string read_text_file(const std::string& data_path) {
    std::cout << "Reading " << data_path << "..." << std::flush;
    auto start = std::chrono::steady_clock::now();
    std::ifstream file(data_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open " + data_path);
    }
    std::string text((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
    file.close();
    auto end = std::chrono::steady_clock::now();
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
        auto start = std::chrono::steady_clock::now();
        tokenizer.load(cache_file);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << " done (" << ms << "ms)" << std::endl;
    } else {
        std::cout << "Training new tokenizer..." << std::flush;
        auto start = std::chrono::steady_clock::now();
        tokenizer.train(text);
        auto end = std::chrono::steady_clock::now();
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
    auto start = std::chrono::steady_clock::now();
    tokens = tokenizer.encode(text);
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << " " << tokens.size() << " tokens (" << ms << "ms)" << std::endl;
}

// A generation prompt as both display text and token ids. Held-out prompts
// keep the ids they were drawn with: the tokenizer drops whitespace, so
// decode-then-encode would not give back the same tokens.
struct Prompt {
    std::string text;
    std::vector<int> tokens;
};

void generate_samples(GPTModel& model, const BPETokenizer& tokenizer,
                      const std::vector<Prompt>& prompts) {
    utils::print_section("Generating Samples");

    TextGen generator(model, &tokenizer);

    std::cout << "\n--- Greedy Decoding ---\n" << std::endl;
    for (const auto& prompt : prompts) {
        std::cout << "Prompt: \"" << prompt.text << "\"" << std::endl;
        std::string generated = generator.generate_greedy(prompt.tokens, 150);
        std::cout << generated << std::endl;
        std::cout << std::string(40, '-') << "\n" << std::endl;
    }

    std::cout << "\n--- Sampling (temp=0.8) ---\n" << std::endl;
    for (const auto& prompt : prompts) {
        std::cout << "Prompt: \"" << prompt.text << "\"" << std::endl;
        std::string generated = generator.generate_sample(prompt.tokens, 0.8f, 150);
        std::cout << generated << std::endl;
        std::cout << std::string(40, '-') << "\n" << std::endl;
    }
}

const char* const kDefaultCorpus = "data/shakespeare.txt";

bool is_default_corpus(const std::string& corpus_path) {
    return corpus_path == kDefaultCorpus;
}

Prompt prompt_from_text(const BPETokenizer& tokenizer, const std::string& text) {
    return {text, tokenizer.encode(text)};
}

// Prompts for end-of-run samples and for `generate` without a prompt.
// Shakespeare keeps its speaker tags, which the model learns to continue in
// character. Any other corpus has no known structure, so the prompts are the
// opening tokens of `count` held-out windows spread evenly across the val
// split: text the model has not trained on, in the corpus's own style.
std::vector<Prompt> sample_prompts(const std::string& corpus_path, const BPETokenizer& tokenizer,
                                   const Dataset& val, size_t count = 3) {
    if (is_default_corpus(corpus_path)) {
        return {prompt_from_text(tokenizer, "ROMEO:\n"),
                prompt_from_text(tokenizer, "JULIET:\n"),
                prompt_from_text(tokenizer, "First Citizen:\n")};
    }
    constexpr size_t kPromptTokens = 6;
    std::vector<Prompt> prompts;
    const size_t windows = val.size();
    for (size_t k = 0; k < count && windows > 0; k++) {
        // Midpoints of `count` equal slices of the split.
        const size_t index = windows * (2 * k + 1) / (2 * count);
        std::vector<int> ids = val.get_item(index).first;
        ids.resize(std::min(ids.size(), kPromptTokens));
        prompts.push_back({tokenizer.decode(ids), ids});
    }
    return prompts;
}

// The default corpus keeps its historical cache name so existing caches
// and checkpoints stay valid; other corpora get corpus-derived names.
std::string tokenizer_cache_prefix(const std::string& corpus_path) {
    if (is_default_corpus(corpus_path)) return "tokenizer";
    return corpus_path + ".tokenizer";
}

std::string token_bin_path(const std::string& corpus_path, int vocab_size,
                           const std::string& split) {
    return corpus_path + "." + std::to_string(vocab_size) + "." + split + ".bin";
}

// BPE merge learning scans every unique word once per merge, so its cost
// grows with corpus size for no statistical benefit: token frequencies
// converge long before 32MB. `prepare` trains on a prefix sample this size.
constexpr size_t kTokenizerSampleBytes = 32ull * 1024 * 1024;

// Inference-time tokenizer loading: the cache must exist (train/prepare
// created it), so the corpus text - possibly gigabytes - is never read.
// Falls back to training from text only for small unprepared corpora; a
// large one would take hours and, unlike `prepare`, would train on the
// full text rather than its sample, giving a different tokenizer.
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
    struct stat st;
    if (::stat(corpus_path.c_str(), &st) == 0
        && static_cast<size_t>(st.st_size) > kTokenizerSampleBytes) {
        throw std::runtime_error("No tokenizer cache " + cache_file + " for a "
                                 + std::to_string(st.st_size >> 20) + "MB corpus; run: "
                                 "./build/grad prepare " + corpus_path + " "
                                 + std::to_string(vocab_size));
    }
    std::string text = read_text_file(corpus_path);
    load_tokenizer(text, tokenizer_cache_prefix(corpus_path), vocab_size, tokenizer);
}

// Vocab for an inference command: the one the user gave, else the
// checkpoint's own. A mismatch is an error before any tokenizer is
// loaded, since a missing cache for a wrong vocab would otherwise start
// BPE training on the corpus.
int resolve_vocab(std::optional<int> requested, const GPTModel& model) {
    if (requested && *requested != model.getVocabSize()) {
        throw std::runtime_error("vocab " + std::to_string(*requested)
                                 + " does not match the checkpoint's vocab of "
                                 + std::to_string(model.getVocabSize())
                                 + " (omit it to use the checkpoint's)");
    }
    return model.getVocabSize();
}

void require_matching_vocab(const BPETokenizer& tokenizer, const GPTModel& model) {
    if (tokenizer.getCurrentVocabSize() != model.getVocabSize()) {
        throw std::runtime_error("tokenizer has " + std::to_string(tokenizer.getCurrentVocabSize())
                                 + " tokens but the checkpoint expects "
                                 + std::to_string(model.getVocabSize())
                                 + "; the cache was built for a different vocab or corpus");
    }
}

// Held-out prompts for `generate` on a corpus other than the default: the
// val token file when the corpus was prepared, else the in-memory 95/5
// split the trainer uses.
std::vector<Prompt> held_out_prompts(const std::string& corpus_path, int vocab_size,
                                     const BPETokenizer& tokenizer) {
    constexpr int kWindow = 6;
    const std::string val_bin = token_bin_path(corpus_path, vocab_size, "val");
    if (tokenfile::exists(val_bin)) {
        return sample_prompts(corpus_path, tokenizer, MappedTokenDataset(val_bin, kWindow, kWindow));
    }
    std::vector<int> tokens = tokenizer.encode(read_text_file(corpus_path));
    const size_t split = tokens.size() * 95 / 100;
    return sample_prompts(corpus_path, tokenizer,
                          TextDataset(std::vector<int>(tokens.begin() + split, tokens.end()),
                                      kWindow, kWindow));
}

// An empty prompt means "pick one": the first speaker tag for the default
// corpus, a held-out opening otherwise.
int run_generation(const std::string& checkpoint_path, const std::string& prompt,
                   const std::string& corpus_path, std::optional<int> requested_vocab) {
    std::cout << "\ngrad.cpp Generation\n" << std::endl;

    try {
        utils::print_section("Loading Model");
        GPTModel model = GPTModel::load(checkpoint_path);
        const int vocab_size = resolve_vocab(requested_vocab, model);

        BPETokenizer tokenizer(vocab_size);
        load_tokenizer_for_inference(corpus_path, vocab_size, tokenizer);
        require_matching_vocab(tokenizer, model);

        const Prompt chosen = !prompt.empty() ? prompt_from_text(tokenizer, prompt)
            : is_default_corpus(corpus_path) ? prompt_from_text(tokenizer, "ROMEO:\n")
            : held_out_prompts(corpus_path, vocab_size, tokenizer).at(0);

        TextGen generator(model, &tokenizer);

        std::cout << "\n--- Greedy Decoding ---\n" << std::endl;
        std::cout << "Prompt: \"" << chosen.text << "\"" << std::endl;
        std::cout << generator.generate_greedy(chosen.tokens, 150) << std::endl;

        std::cout << "\n--- Sampling (temp=0.8) ---\n" << std::endl;
        std::cout << "Prompt: \"" << chosen.text << "\"" << std::endl;
        std::cout << generator.generate_sample(chosen.tokens, 0.8f, 150) << std::endl;

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
             const std::string& corpus_path, std::optional<int> requested_vocab) {
    std::cout << "\ngrad.cpp Chat\n" << std::endl;

    try {
        utils::print_section("Loading Model");
        GPTModel model = GPTModel::load(checkpoint_path);
        const int vocab_size = resolve_vocab(requested_vocab, model);

        BPETokenizer tokenizer(vocab_size);
        load_tokenizer_for_inference(corpus_path, vocab_size, tokenizer);
        require_matching_vocab(tokenizer, model);
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

// Scores a checkpoint on the held-out split and on an equal-sized random
// sample of training windows, both in non-overlapping windows. The trainer's
// in-loop validation reads only the first max_eval_batches of the split, which
// is fine for tracking a run but not for reporting one; this is the number to
// report. max_batches = 0 walks the whole val split; otherwise both splits are
// sampled uniformly, so a capped run still covers the split rather than its
// first few stories. The train-side figure separates generalization from
// distribution shift between the two splits.
int run_eval(const std::string& checkpoint_path, const std::string& corpus_path,
             std::optional<int> requested_vocab, int seq_length, int max_batches) {
    std::cout << "\ngrad.cpp Evaluation\n" << std::endl;

    try {
        utils::print_section("Loading Model");
        GPTModel model = GPTModel::load(checkpoint_path);
        const int vocab_size = resolve_vocab(requested_vocab, model);

        const std::string train_bin = token_bin_path(corpus_path, vocab_size, "train");
        const std::string val_bin = token_bin_path(corpus_path, vocab_size, "val");
        if (!tokenfile::exists(train_bin) || !tokenfile::exists(val_bin)) {
            throw std::runtime_error("eval needs pre-tokenized files; run: ./build/grad prepare "
                                     + corpus_path + " " + std::to_string(vocab_size));
        }
        if (seq_length > model.getMaxLen()) {
            throw std::runtime_error("seq length exceeds the model's context ("
                                     + std::to_string(model.getMaxLen()) + ")");
        }

        constexpr int kBatchSize = 8;
        auto val = std::make_shared<MappedTokenDataset>(val_bin, seq_length, seq_length);
        auto train = std::make_shared<MappedTokenDataset>(train_bin, seq_length, seq_length);
        if (val->vocabSize() != model.getVocabSize()) {
            throw std::runtime_error("token files and checkpoint disagree on vocab size");
        }

        constexpr unsigned kSeed = 20260921;
        DataLoader val_loader(val, kBatchSize, /*shuffle=*/max_batches > 0, kSeed);
        const int val_batches = max_batches > 0
            ? std::min<int>(max_batches, static_cast<int>(val_loader.num_batches()))
            : static_cast<int>(val_loader.num_batches());
        DataLoader train_loader(train, kBatchSize, /*shuffle=*/true, kSeed + 1);

        auto report = [](const char* split, double loss, long tokens, double seconds) {
            std::cout << std::left << std::setw(8) << split << std::right << std::fixed
                      << " loss " << std::setprecision(4) << loss
                      << "  perplexity " << std::setprecision(3) << std::exp(loss)
                      << "  (" << tokens << " tokens, " << std::setprecision(0) << seconds
                      << "s)" << std::defaultfloat << std::endl;
        };
        auto timed = [&](DataLoader& loader) {
            const auto start = std::chrono::steady_clock::now();
            const double loss = training::mean_loss(model, loader, val_batches);
            const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
            return std::make_pair(loss, elapsed.count());
        };

        utils::print_section("Scoring");
        const long tokens = static_cast<long>(std::min<size_t>(
            static_cast<size_t>(val_batches) * kBatchSize, val->size())) * seq_length;
        const auto [val_loss, val_s] = timed(val_loader);
        report("val", val_loss, tokens, val_s);
        const auto [train_loss, train_s] = timed(train_loader);
        report("train", train_loss, static_cast<long>(val_batches) * kBatchSize * seq_length,
               train_s);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}

struct BenchmarkOptions {
    int steps = 20;
    int warmup = 3;
    int trials = 5;
    std::string json_path;
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2.0;
}

std::string benchmark_timestamp() {
    std::time_t now = std::time(nullptr);
    std::tm utc {};
#ifdef _WIN32
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

void write_benchmark_json(const BenchmarkOptions& options,
                          const std::vector<double>& train_steps_per_s,
                          const std::vector<double>& train_tokens_per_s,
                          const std::vector<double>& generation_tokens_per_s,
                          size_t parameter_count,
                          size_t peak_memory_mb) {
    if (options.json_path.empty()) return;
    std::ofstream out(options.json_path);
    if (!out) throw std::runtime_error("Cannot write benchmark JSON: " + options.json_path);

    auto array = [&](const std::vector<double>& values) {
        out << "[";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i) out << ", ";
            out << std::fixed << std::setprecision(6) << values[i];
        }
        out << "]";
    };

    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"timestamp_utc\": \"" << benchmark_timestamp() << "\",\n"
        << "  \"engine\": \"grad.cpp\",\n"
        << "  \"version\": \"" << GRAD_VERSION << "\",\n"
        << "  \"git_sha\": \"" << GRAD_GIT_SHA << "\",\n"
        << "  \"build_type\": \"" << GRAD_BUILD_TYPE << "\",\n"
        << "  \"compiler\": \"" << GRAD_COMPILER << "\",\n"
        << "  \"system\": \"" << GRAD_SYSTEM << "\",\n"
        << "  \"metal_available\": " << (metalgpu::available() ? "true" : "false") << ",\n"
        << "  \"metal_fp16\": " << (metalgpu::fp16_active() ? "true" : "false") << ",\n"
        << "  \"parameters\": " << parameter_count << ",\n"
        << "  \"config\": {\"vocab\": 5000, \"d_model\": 512, "
           "\"layers\": 6, \"heads\": 8, \"sequence\": 96, \"batch\": 8},\n"
        << "  \"protocol\": {\"warmup_steps\": " << options.warmup
        << ", \"steps_per_trial\": " << options.steps
        << ", \"trials\": " << options.trials << "},\n"
        << "  \"training_steps_per_second\": ";
    array(train_steps_per_s);
    out << ",\n  \"training_tokens_per_second\": ";
    array(train_tokens_per_s);
    out << ",\n  \"generation_tokens_per_second\": ";
    array(generation_tokens_per_s);
    out << ",\n"
        << "  \"median_training_steps_per_second\": " << median(train_steps_per_s) << ",\n"
        << "  \"median_training_tokens_per_second\": " << median(train_tokens_per_s) << ",\n"
        << "  \"median_generation_tokens_per_second\": " << median(generation_tokens_per_s) << ",\n"
        << "  \"peak_rss_mb\": " << peak_memory_mb << "\n"
        << "}\n";
    if (!out.good()) throw std::runtime_error("Failed while writing benchmark JSON: " + options.json_path);
}

// Repeatable performance benchmark: initialization and tokenization stay
// outside the timed region; each reported number is the median of trials.
int run_benchmark(const BenchmarkOptions& options) {
    std::cout << "\ngrad.cpp Benchmark\n" << std::endl;

    try {
        const int vocab_size = 5000;
        if (options.steps < 1 || options.warmup < 0 || options.trials < 1) {
            throw std::invalid_argument("bench steps/trials must be positive and warmup non-negative");
        }

        BPETokenizer tokenizer(vocab_size);
        std::string text = read_text_file("data/shakespeare.txt");
        load_tokenizer(text, "tokenizer", vocab_size, tokenizer);
        std::vector<int> tokens = tokenizer.encode(text);

        const int d_model = 512, num_layers = 6, num_heads = 8;
        const int max_len = 1024, seq_length = 96, batch_size = 8;

        GPTModel model(vocab_size, d_model, num_layers, num_heads, max_len, 0.1f);
        auto params = model.getAllParameters();
        size_t parameter_count = 0;
        for (const auto& param : params) parameter_count += param->getData().numel();
        AdamOptimizer optimizer(params, 3e-4f, 0.9f, 0.999f, 1e-8f, 0.0f);

        auto dataset = std::make_shared<TextDataset>(tokens, seq_length);
        DataLoader loader(dataset, batch_size, true);

        utils::print_section("Training throughput");
        std::cout << "Config: d_model=" << d_model << " layers=" << num_layers
                  << " heads=" << num_heads << " seq=" << seq_length
                  << " batch=" << batch_size << " params=" << parameter_count << std::endl;

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

        std::cout << "Protocol: " << options.warmup << " warmup, " << options.trials
                  << " trials x " << options.steps << " steps\n"
                  << "Build: grad.cpp " << GRAD_VERSION << " (" << GRAD_GIT_SHA << "), "
                  << GRAD_BUILD_TYPE << ", " << GRAD_COMPILER << std::endl;

        for (int i = 0; i < options.warmup; i++) run_step();

        std::vector<double> train_steps_per_s;
        std::vector<double> train_tokens_per_s;
        for (int trial = 0; trial < options.trials; ++trial) {
            auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < options.steps; i++) run_step();
            auto end = std::chrono::steady_clock::now();
            double train_s = std::chrono::duration<double>(end - start).count();
            double steps_per_s = options.steps / train_s;
            double tokens_per_s = steps_per_s * batch_size * seq_length;
            train_steps_per_s.push_back(steps_per_s);
            train_tokens_per_s.push_back(tokens_per_s);
            std::cout << "  trial " << (trial + 1) << ": " << std::fixed << std::setprecision(3)
                      << train_s << "s, " << std::setprecision(2) << steps_per_s
                      << " steps/s, " << std::setprecision(0) << tokens_per_s << " tok/s"
                      << std::endl;
        }
        std::cout << "  median: " << std::fixed << std::setprecision(2)
                  << median(train_steps_per_s) << " steps/s, " << std::setprecision(0)
                  << median(train_tokens_per_s) << " tok/s" << std::endl;

        utils::print_section("Generation throughput");
        const int gen_tokens = 64;
        TextGen generator(model, &tokenizer);
        auto prompt = tokenizer.encode("ROMEO:\n");

        std::vector<double> generation_tokens_per_s;
        for (int trial = 0; trial < options.trials; ++trial) {
            auto start = std::chrono::steady_clock::now();
            generator.generate_sample(prompt, 0.8f, gen_tokens);
            auto end = std::chrono::steady_clock::now();
            double gen_s = std::chrono::duration<double>(end - start).count();
            double tokens_per_s = gen_tokens / gen_s;
            generation_tokens_per_s.push_back(tokens_per_s);
            std::cout << "  trial " << (trial + 1) << ": " << std::fixed << std::setprecision(1)
                      << tokens_per_s << " tok/s" << std::endl;
        }
        std::cout << "  median: " << std::fixed << std::setprecision(1)
                  << median(generation_tokens_per_s) << " tok/s" << std::endl;

        size_t peak_memory_mb = utils::get_peak_memory_mb();
        std::cout << "\nPeak RSS: " << peak_memory_mb << " MB" << std::endl;
        write_benchmark_json(options, train_steps_per_s, train_tokens_per_s,
                             generation_tokens_per_s, parameter_count, peak_memory_mb);
        if (!options.json_path.empty()) {
            std::cout << "Benchmark JSON: " << options.json_path << std::endl;
        }
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
    std::cout << "\ngrad.cpp Prepare\n" << std::endl;

    try {
        utils::print_section("Tokenizing corpus");
        std::string text = read_text_file(corpus_path);
        BPETokenizer tokenizer(vocab_size);

        // Train the tokenizer on a prefix sample (cut at a word boundary;
        // see kTokenizerSampleBytes), then encode the full corpus with it.
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
        auto start = std::chrono::steady_clock::now();
        std::vector<int> tokens = tokenizer.encode(text);
        auto end = std::chrono::steady_clock::now();
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
                  << "\n\nTrain with: ./build/grad train " << corpus_path
                  << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}

// Model/training presets. "small" is the original 22M-param Shakespeare
// config; "fast" is the CI smoke test.
//
// "medium" (~70M params) is memory-bound, not compute-bound: one step at
// batch 16 / seq 256 once peaked past 10GB of phys footprint and took a
// 16GB machine down (measured 2026-07-18, BENCHMARKS.md #8). Micro-batch
// 8 with grad_accum 4 gives the optimizer an effective batch of 32 at a
// ~4GB peak (lazy grads + backward retirement, BENCHMARKS.md #9). The
// micro-batch is also measured, not assumed: 16 x accum 2 - same
// effective batch, FFN matmuls above the Metal crossover - ran 32%
// slower per token and 3GB fatter, with or without the GPU. At 40000
// steps the run consumes 40000*32*256 = 327M tokens, about one epoch of
// the TinyStories train split.
// "modern" is "medium" with the Llama-style block (RMSNorm, RoPE, SwiGLU;
// see gpt_model.h) at the same parameter count and training budget, so the
// two runs A/B the architecture and nothing else. Its checkpoints get a
// "_modern" prefix so both lineages can coexist on one corpus.
struct Preset {
    const char* name;
    int vocab_size;
    int d_model, num_layers, num_heads;
    int max_len, seq_length, batch_size, grad_accum;
    float learning_rate;
    float dropout;
    int warmup_steps, num_steps;
    int checkpoint_interval, eval_interval;
    int max_eval_batches;  // 0 = evaluate the whole val set
    bool modern;           // GPTArch::Modern
};

// small's step count is set by measurement: on the 254K-token Shakespeare
// corpus, val perplexity bottoms around step 6000 and rises after (a 22M
// model memorizes a corpus that small). 8000 steps lets the cosine
// schedule finish near the minimum instead of training 8x past it.
//
// Dropout is per preset because it answers a per-corpus question. small
// repeats its 254K-token corpus many times over and memorizes it, so 0.1
// earns its keep. medium/modern see 327M tokens of a 363M-token corpus,
// under one epoch: there is no repeated exposure to regularize against,
// and the mask generation was the largest non-BLAS cost in the profile
// (13% of compute). Single-epoch pretraining runs use 0 for this reason.
const Preset kPresets[] = {
    {"fast",        500,   128, 2,  4,  1024, 64,  4, 1,  3e-4f, 0.1f, 10,   50,    2500, 25,  0,  false},
    {"fast-modern", 500,   128, 2,  4,  1024, 64,  4, 1,  3e-4f, 0.1f, 10,   50,    2500, 25,  0,  true},
    {"small",       5000,  512, 6,  8,  1024, 96,  8, 1,  3e-4f, 0.1f, 500,  8000,  2500, 250, 0,  false},
    {"medium",      16000, 768, 8,  12, 1024, 256, 8, 4,  3e-4f, 0.0f, 1000, 40000, 4000, 500, 32, false},
    {"modern",      16000, 768, 8,  12, 1024, 256, 8, 4,  3e-4f, 0.0f, 1000, 40000, 4000, 500, 32, true},
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

// init_arg: "" trains from scratch; "resume" continues an interrupted run
// from <prefix>_resume_model.bin / _resume_state.bin (optimizer state and
// schedule position included); any other value is a checkpoint path to
// warm-start from - weights only, fresh optimizer and schedule.
int run_training(const Preset& preset, const std::string& corpus_path,
                 const std::string& init_arg) {
    std::cout << "\ngrad.cpp Training (" << preset.name << ")\n" << std::endl;

    try {
        const int vocab_size = preset.vocab_size;
        const int num_steps = preset.num_steps;
        const int seq_length = preset.seq_length;
        const bool fast_mode = std::string(preset.name).rfind("fast", 0) == 0;
        const GPTArch arch = preset.modern ? GPTArch::Modern : GPTArch::GPT2;

        const std::string prefix = checkpoint_stem(corpus_path)
                                 + (preset.modern ? "_modern" : "")
                                 + (fast_mode ? "_fast" : "");
        const bool resume = (init_arg == "resume");
        const std::string warm_start_path = resume ? "" : init_arg;

        std::optional<int> resume_next_step;
        if (resume) {
            resume_next_step = training::peek_resume_step(prefix + "_resume_state.bin");
            if (!resume_next_step) {
                throw std::runtime_error("No resume state found ("
                    + prefix + "_resume_state.bin); start a run first");
            }
            if (*resume_next_step >= num_steps) {
                std::cout << "Run already completed all " << num_steps
                          << " steps; nothing to resume." << std::endl;
                return 0;
            }
        }

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
                                         + cache_file + "); run: ./build/grad prepare "
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
        config.grad_accum = preset.grad_accum;
        config.learning_rate = preset.learning_rate;
        config.dropout = preset.dropout;
        config.warmup_steps = preset.warmup_steps;
        config.num_steps = num_steps;
        config.checkpoint_interval = preset.checkpoint_interval;
        config.checkpoint_prefix = prefix;
        config.eval_interval = preset.eval_interval;
        config.max_eval_batches = preset.max_eval_batches;

        auto start = std::chrono::steady_clock::now();
        GPTModel model = [&]() -> GPTModel {
            if (resume) return GPTModel::load(prefix + "_resume_model.bin");
            if (!warm_start_path.empty()) {
                std::cout << "Warm start from " << warm_start_path
                          << " (weights only, fresh optimizer)" << std::endl;
                return GPTModel::load(warm_start_path);
            }
            return GPTModel(config.vocab_size, config.d_model, config.num_layers,
                            config.num_heads, config.max_len, config.dropout, arch);
        }();
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (model.getVocabSize() != config.vocab_size
            || model.getDModel() != config.d_model
            || model.getNumLayers() != config.num_layers
            || model.getNumHeads() != config.num_heads
            || model.getArch() != arch) {
            throw std::runtime_error("Checkpoint architecture does not match preset '"
                                     + std::string(preset.name) + "'");
        }

        auto params = model.getAllParameters();
        size_t total_params = 0;
        for (const auto& p : params) total_params += p->getData().numel();

        std::cout << "Model initialized (" << ms << "ms)" << std::endl;
        std::cout << "Parameters: " << (static_cast<double>(total_params) / 1e6) << "M" << std::endl;

        // The training loader samples windows with replacement, so any run
        // that starts from existing weights must not repeat the seed those
        // weights were trained with - it would replay the exact batch
        // sequence the checkpoint already saw. Resumes perturb the seed by
        // their step position, warm starts by the checkpoint path.
        unsigned int loader_seed = 42;
        if (resume) {
            loader_seed = 42u + static_cast<unsigned int>(*resume_next_step);
        } else if (!warm_start_path.empty()) {
            loader_seed = static_cast<unsigned int>(
                std::hash<std::string>{}(warm_start_path));
        }
        DataLoader loader(dataset, config.batch_size, true, loader_seed);
        DataLoader val_loader(val_dataset, config.batch_size, false);

        std::cout << "Dataset: " << dataset->size() << " train / "
                  << val_dataset->size() << " val sequences\n" << std::endl;

        training::Trainer trainer(config, model, loader, &val_loader);
        if (resume && !trainer.load_resume_state()) {
            throw std::runtime_error("Failed to load resume state ("
                                     + prefix + "_resume_state.bin)");
        }

        if (!trainer.train()) {
            const bool via_train_fast = std::string(preset.name) == "fast";
            std::cout << "\nResume with: ./build/grad "
                      << (via_train_fast ? "train-fast " + corpus_path
                                         : "train " + corpus_path + " " + preset.name)
                      << " resume\n" << std::endl;
            return 0;
        }

        generate_samples(model, tokenizer, sample_prompts(corpus_path, tokenizer, *val_dataset));

        std::cout << "\nTraining Complete!\n" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}

// Whole-string integer parse: "16k", "", and "16000x" are errors rather
// than atoi's silent 16 / 0 / 16000.
int parse_int(const char* text, const char* what, int min_value) {
    int value = 0;
    const char* end = text + std::strlen(text);
    const auto [ptr, ec] = std::from_chars(text, end, value);
    if (ec != std::errc() || ptr != end || ptr == text) {
        throw std::invalid_argument(std::string(what) + " must be an integer, got '" + text + "'");
    }
    if (value < min_value) {
        throw std::invalid_argument(std::string(what) + " must be >= " + std::to_string(min_value)
                                    + ", got " + text);
    }
    return value;
}

std::optional<int> optional_vocab(int argc, char* argv[], int index) {
    if (argc <= index) return std::nullopt;
    return parse_int(argv[index], "vocab", 1);
}

int run_cli(int argc, char* argv[]) {
    std::string mode = (argc > 1) ? argv[1] : "";

    const std::string default_corpus = kDefaultCorpus;

    if (mode == "train") {
        std::string corpus = (argc > 2) ? argv[2] : default_corpus;
        std::string preset_name = (argc > 3) ? argv[3] : "small";
        std::string init_arg = (argc > 4) ? argv[4] : "";
        const Preset* preset = find_preset(preset_name);
        if (!preset || preset_name == "fast") {
            std::cerr << "Unknown preset '" << preset_name << "' (available: small, medium, modern)" << std::endl;
            return 1;
        }
        return run_training(*preset, corpus, init_arg);
    }
    if (mode == "train-fast") {
        return run_training(*find_preset("fast"), (argc > 2) ? argv[2] : default_corpus,
                            (argc > 3) ? argv[3] : "");
    }
    if (mode == "prepare") {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " prepare <corpus.txt> [vocab_size]" << std::endl;
            return 1;
        }
        const int vocab = (argc > 3) ? parse_int(argv[3], "vocab", 1) : 5000;
        return run_prepare(argv[2], vocab);
    }
    if (mode == "generate") {
        std::string checkpoint = (argc > 2) ? argv[2] : "shakespeare_final.bin";
        std::string prompt = (argc > 3) ? argv[3] : "";
        std::string corpus = (argc > 4) ? argv[4] : default_corpus;
        return run_generation(checkpoint, prompt, corpus, optional_vocab(argc, argv, 5));
    }
    if (mode == "eval") {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0]
                      << " eval <ckpt> <corpus.txt> [vocab] [seq] [max_batches]" << std::endl;
            return 1;
        }
        const int seq = (argc > 5) ? parse_int(argv[5], "seq", 1) : 256;
        const int max_batches = (argc > 6) ? parse_int(argv[6], "max_batches", 0) : 0;
        return run_eval(argv[2], argv[3], optional_vocab(argc, argv, 4), seq, max_batches);
    }
    if (mode == "bench") {
        BenchmarkOptions options;
        bool positional_steps_seen = false;
        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            auto require_value = [&](const char* flag) -> std::string {
                if (++i >= argc) throw std::invalid_argument(std::string("Missing value for ") + flag);
                return argv[i];
            };
            try {
                if (arg == "--steps") options.steps = std::stoi(require_value("--steps"));
                else if (arg == "--warmup") options.warmup = std::stoi(require_value("--warmup"));
                else if (arg == "--trials") options.trials = std::stoi(require_value("--trials"));
                else if (arg == "--json") options.json_path = require_value("--json");
                else if (!positional_steps_seen && !arg.empty() && arg[0] != '-') {
                    options.steps = std::stoi(arg);
                    positional_steps_seen = true;
                } else {
                    throw std::invalid_argument("Unknown bench argument: " + arg);
                }
            } catch (const std::exception& e) {
                std::cerr << "Benchmark argument error: " << e.what() << std::endl;
                return 1;
            }
        }
        return run_benchmark(options);
    }
    if (mode == "watch") {
        // Argument is a run prefix ("tinystories_modern"), a metrics path,
        // or nothing - then the most recently modified *_metrics.csv in
        // the working directory (i.e. whatever is training right now).
        std::string target = (argc > 2) ? argv[2] : "";
        bool once = false;
        for (int i = 2; i < argc; i++) {
            if (std::string(argv[i]) == "--once") { once = true; if (target == argv[i]) target = ""; }
        }
        std::string csv;
        if (!target.empty() && target != "--once") {
            csv = (target.size() > 4 && target.substr(target.size() - 4) == ".csv")
                ? target : target + "_metrics.csv";
        } else {
            csv = utils::newest_metrics_csv(".");
            if (csv.empty()) {
                std::cerr << "No *_metrics.csv found; start a training run first "
                          << "(runs on this build log metrics automatically)." << std::endl;
                return 1;
            }
        }
        return utils::run_dashboard(csv, once);
    }
    if (mode == "chat") {
        std::string checkpoint = (argc > 2) ? argv[2] : "shakespeare_final.bin";
        std::string corpus = (argc > 3) ? argv[3] : default_corpus;
        return run_chat(checkpoint, corpus, optional_vocab(argc, argv, 4));
    }

    std::cerr << "Usage: " << argv[0] << " <mode>\n"
              << "  prepare <corpus.txt> [vocab]         pre-tokenize a corpus to .bin token files\n"
              << "  train [corpus.txt] [small|medium|modern] [ckpt.bin|resume]\n"
              << "      full training run (uses .bin files if present); Ctrl-C saves resume\n"
              << "      state - 'resume' continues an interrupted run, a checkpoint path\n"
              << "      warm-starts from those weights with a fresh schedule\n"
              << "  train-fast [corpus.txt] [ckpt.bin|resume]   tiny config for a quick smoke test\n"
              << "  generate [ckpt] [prompt] [corpus] [vocab]   sample from a saved checkpoint\n"
              << "      an empty prompt (\"\") picks one from the corpus's held-out split\n"
              << "  chat [ckpt] [corpus] [vocab]                interactive prompt/continue REPL\n"
              << "  eval <ckpt> <corpus> [vocab] [seq] [max_batches]\n"
              << "      loss/perplexity on the full val split and a matched train sample\n"
              << "  (generate/chat/eval take the vocab from the checkpoint when it is omitted)\n"
              << "  bench [steps] [--warmup N] [--trials N] [--json path]\n"
              << "                                      benchmark with repeated median trials\n"
              << "  watch [run-prefix]                   live terminal dashboard for a training\n"
              << "      run (loss curves, val track, throughput); defaults to the most recent\n"
              << "      run in this directory - open it in a second terminal while training\n";
    return 1;
}

int main(int argc, char* argv[]) {
    try {
        return run_cli(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
