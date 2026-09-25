// grad bench: repeatable training and generation throughput. Model
// construction and tokenization stay outside the timed region; each
// reported number is the median of repeated trials, and --json writes a
// machine-readable record with build provenance (schema_version 1).

#include "cli/args.h"
#include "commands.h"
#include "common.h"

#include "grad/data/dataloader.h"
#include "grad/data/dataset.h"
#include "grad/transformer/gpt_model.h"
#include "grad/transformer/metal_backend.h"
#include "grad/transformer/optimizer.h"
#include "grad/transformer/text_gen.h"
#include "grad/transformer/variable.h"
#include "grad/utils/metrics.h"
#include "grad/utils/training_utils.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

namespace grad::cli {

namespace {

struct BenchmarkOptions {
    int steps = 20;
    int warmup = 3;
    int trials = 5;
    std::string json_path;
};

// The benchmarked model: the 22M "small" config, pinned here rather than
// read from the preset so BENCHMARKS.md stays comparable if small changes.
struct BenchmarkConfig {
    int vocab_size;
    int d_model;
    int num_layers;
    int num_heads;
    int max_len;
    int seq_length;
    int batch_size;
    float dropout;
};

constexpr BenchmarkConfig kConfig{
    .vocab_size = 5000,
    .d_model = 512,
    .num_layers = 6,
    .num_heads = 8,
    .max_len = 1024,
    .seq_length = 96,
    .batch_size = 8,
    .dropout = 0.1f,
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2.0;
}

std::string benchmark_timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
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
                          size_t parameter_count, size_t peak_memory_mb) {
    if (options.json_path.empty()) return;
    std::ofstream out(options.json_path);
    if (!out) throw std::runtime_error("Cannot write benchmark JSON: " + options.json_path);

    // Leaves the stream in fixed/6 for the medians that follow.
    const auto array = [&](const std::vector<double>& values) {
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
        << "  \"metal_available\": " << (metal::available() ? "true" : "false") << ",\n"
        << "  \"metal_fp16\": " << (metal::fp16_active() ? "true" : "false") << ",\n"
        << "  \"parameters\": " << parameter_count << ",\n"
        << "  \"config\": {\"vocab\": " << kConfig.vocab_size
        << ", \"d_model\": " << kConfig.d_model << ", \"layers\": " << kConfig.num_layers
        << ", \"heads\": " << kConfig.num_heads << ", \"sequence\": " << kConfig.seq_length
        << ", \"batch\": " << kConfig.batch_size << "},\n"
        << "  \"protocol\": {\"warmup_steps\": " << options.warmup
        << ", \"steps_per_trial\": " << options.steps << ", \"trials\": " << options.trials
        << "},\n"
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
    if (!out.good()) {
        throw std::runtime_error("Failed while writing benchmark JSON: " + options.json_path);
    }
}

void benchmark(const BenchmarkOptions& options) {
    std::cout << "\ngrad.cpp Benchmark\n" << std::endl;

    const int vocab_size = kConfig.vocab_size;
    BPETokenizer tokenizer(vocab_size);
    const std::string text = read_text_file(kDefaultCorpus);
    load_tokenizer(text, "tokenizer", vocab_size, tokenizer);
    const std::vector<int> tokens = tokenizer.encode(text);

    const int seq_length = kConfig.seq_length;
    const int batch_size = kConfig.batch_size;

    GPTModel model(vocab_size, kConfig.d_model, kConfig.num_layers, kConfig.num_heads,
                   kConfig.max_len, kConfig.dropout);
    const auto params = model.getAllParameters();
    size_t parameter_count = 0;
    for (const auto& param : params) parameter_count += param->getData().numel();
    AdamOptimizer optimizer(params, 3e-4f, 0.9f, 0.999f, 1e-8f, 0.0f);

    auto dataset = std::make_shared<TextDataset>(tokens, seq_length);
    DataLoader loader(dataset, batch_size, true);

    utils::print_section("Training throughput");
    std::cout << "Config: d_model=" << kConfig.d_model << " layers=" << kConfig.num_layers
              << " heads=" << kConfig.num_heads << " seq=" << seq_length << " batch=" << batch_size
              << " params=" << parameter_count << std::endl;

    const auto run_step = [&]() {
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

    std::cout << "Protocol: " << options.warmup << " warmup, " << options.trials << " trials x "
              << options.steps << " steps\n"
              << "Build: grad.cpp " << GRAD_VERSION << " (" << GRAD_GIT_SHA << "), "
              << GRAD_BUILD_TYPE << ", " << GRAD_COMPILER << std::endl;

    for (int i = 0; i < options.warmup; i++) run_step();

    std::vector<double> train_steps_per_s;
    std::vector<double> train_tokens_per_s;
    for (int trial = 0; trial < options.trials; ++trial) {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < options.steps; i++) run_step();
        const auto end = std::chrono::steady_clock::now();
        const double train_s = std::chrono::duration<double>(end - start).count();
        const double steps_per_s = options.steps / train_s;
        const double tokens_per_s = steps_per_s * batch_size * seq_length;
        train_steps_per_s.push_back(steps_per_s);
        train_tokens_per_s.push_back(tokens_per_s);
        std::cout << "  trial " << (trial + 1) << ": " << std::fixed << std::setprecision(3)
                  << train_s << "s, " << std::setprecision(2) << steps_per_s << " steps/s, "
                  << std::setprecision(0) << tokens_per_s << " tok/s" << std::endl;
    }
    std::cout << "  median: " << std::fixed << std::setprecision(2) << median(train_steps_per_s)
              << " steps/s, " << std::setprecision(0) << median(train_tokens_per_s) << " tok/s"
              << std::endl;

    utils::print_section("Generation throughput");
    constexpr int kGenTokens = 64;
    TextGen generator(model, &tokenizer);
    const auto prompt = tokenizer.encode("ROMEO:\n");

    std::vector<double> generation_tokens_per_s;
    for (int trial = 0; trial < options.trials; ++trial) {
        const auto start = std::chrono::steady_clock::now();
        generator.generate_sample(prompt, 0.8f, kGenTokens);
        const auto end = std::chrono::steady_clock::now();
        const double gen_s = std::chrono::duration<double>(end - start).count();
        const double tokens_per_s = kGenTokens / gen_s;
        generation_tokens_per_s.push_back(tokens_per_s);
        std::cout << "  trial " << (trial + 1) << ": " << std::fixed << std::setprecision(1)
                  << tokens_per_s << " tok/s" << std::endl;
    }
    std::cout << "  median: " << std::fixed << std::setprecision(1)
              << median(generation_tokens_per_s) << " tok/s" << std::endl;

    const size_t peak_memory_mb = utils::get_peak_memory_mb();
    std::cout << "\nPeak RSS: " << peak_memory_mb << " MB" << std::endl;
    write_benchmark_json(options, train_steps_per_s, train_tokens_per_s, generation_tokens_per_s,
                         parameter_count, peak_memory_mb);
    if (!options.json_path.empty()) {
        std::cout << "Benchmark JSON: " << options.json_path << std::endl;
    }
}

}  // namespace

int run_bench(const Invocation& invocation) {
    BenchmarkOptions options;

    Command cmd(invocation.usage_name(), std::string(invocation.summary));
    cmd.describe(
        "Times optimizer steps of the 22M 'small' model on Tiny Shakespeare, then "
        "64-token sampled generation, reporting the median of --trials windows. "
        "Run it on an otherwise idle machine.");
    cmd.optional("steps", options.steps, "timed optimizer steps per trial").at_least(1);
    cmd.option("--warmup", options.warmup, "untimed steps before the first trial").at_least(0);
    cmd.option("--trials", options.trials, "timed windows; the median is reported").at_least(1);
    cmd.option("--json", options.json_path, "also write the results as JSON").metavar("PATH");
    if (cmd.parse(invocation.args) == ParseResult::HelpShown) return 0;

    benchmark(options);
    return 0;
}

}  // namespace grad::cli
