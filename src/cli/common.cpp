#include "common.h"

#include "utils/metrics.h"

#include <sys/stat.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace cli {

namespace {

bool file_exists(const std::string& path) {
    return std::ifstream(path).good();
}

// Inference-time tokenizer loading: the cache must exist (train/prepare
// created it), so the corpus text - possibly gigabytes - is never read.
// Falls back to training from text only for small unprepared corpora; a
// large one would take hours and, unlike `prepare`, would train on the
// full text rather than its sample, giving a different tokenizer.
void load_tokenizer_for_inference(const std::string& corpus_path, int vocab_size,
                                  BPETokenizer& tokenizer) {
    const std::string cache_file = tokenizer_cache_path(corpus_path, vocab_size);
    if (file_exists(cache_file)) {
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
    const std::string text = read_text_file(corpus_path);
    load_tokenizer(text, tokenizer_cache_prefix(corpus_path), vocab_size, tokenizer);
}

void require_matching_vocab(const BPETokenizer& tokenizer, const GPTModel& model) {
    if (tokenizer.getCurrentVocabSize() != model.getVocabSize()) {
        throw std::runtime_error("tokenizer has " + std::to_string(tokenizer.getCurrentVocabSize())
                                 + " tokens but the checkpoint expects "
                                 + std::to_string(model.getVocabSize())
                                 + "; the cache was built for a different vocab or corpus");
    }
}

}  // namespace

bool is_default_corpus(const std::string& corpus_path) {
    return corpus_path == kDefaultCorpus;
}

std::string tokenizer_cache_prefix(const std::string& corpus_path) {
    if (is_default_corpus(corpus_path)) return "tokenizer";
    return corpus_path + ".tokenizer";
}

std::string tokenizer_cache_path(const std::string& corpus_path, int vocab_size) {
    return tokenizer_cache_prefix(corpus_path) + "_" + std::to_string(vocab_size) + ".cache";
}

std::string token_bin_path(const std::string& corpus_path, int vocab_size,
                           const std::string& split) {
    return corpus_path + "." + std::to_string(vocab_size) + "." + split + ".bin";
}

std::string read_text_file(const std::string& path) {
    std::cout << "Reading " << path << "..." << std::flush;
    const Stopwatch timer;
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open " + path);
    }
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::cout << " " << (text.size() / 1024) << "KB (" << timer.ms() << "ms)" << std::endl;
    return text;
}

void load_tokenizer(const std::string& text, const std::string& cache_prefix, int vocab_size,
                    BPETokenizer& tokenizer) {
    const std::string cache_file = cache_prefix + "_" + std::to_string(vocab_size) + ".cache";
    if (file_exists(cache_file)) {
        std::cout << "Loading tokenizer from cache..." << std::flush;
        const Stopwatch timer;
        tokenizer.load(cache_file);
        std::cout << " done (" << timer.ms() << "ms)" << std::endl;
    } else {
        std::cout << "Training new tokenizer..." << std::flush;
        const Stopwatch timer;
        tokenizer.train(text);
        std::cout << " done (" << timer.ms() << "ms)" << std::endl;
        tokenizer.save(cache_file);
        std::cout << "Cached to " << cache_file << std::endl;
    }
    std::cout << "Vocab size: " << tokenizer.getCurrentVocabSize() << std::endl;
}

Prompt prompt_from_text(const BPETokenizer& tokenizer, const std::string& text) {
    return {text, tokenizer.encode(text)};
}

std::vector<Prompt> sample_prompts(const std::string& corpus_path, const BPETokenizer& tokenizer,
                                   const Dataset& val, size_t count) {
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

void add_sampling_options(Command& cmd, SamplingOptions& options, const std::string& greedy_help) {
    cmd.option("--max-tokens", options.max_tokens, "tokens to generate per continuation")
        .at_least(1);
    cmd.option("--temperature", options.temperature,
               "sampling temperature; lower is more conservative")
        .positive();
    cmd.option("--top-k", options.top_k, "sample from the k most likely tokens only; 0 is off")
        .at_least(0);
    cmd.option("--top-p", options.top_p,
               "sample from the smallest set of tokens whose probability reaches p; 1 is off")
        .positive()
        .at_most(1);
    cmd.option("--repetition-penalty", options.repetition_penalty,
               "divides the logits of the last 50 tokens; 1 is off")
        .positive();
    cmd.flag("--greedy", options.greedy, greedy_help);
}

GPTModel load_checkpoint(const std::string& checkpoint_path, std::optional<int> requested_vocab) {
    utils::print_section("Loading Model");
    GPTModel model = GPTModel::load(checkpoint_path);
    if (requested_vocab && *requested_vocab != model.getVocabSize()) {
        throw std::runtime_error("vocab " + std::to_string(*requested_vocab)
                                 + " does not match the checkpoint's vocab of "
                                 + std::to_string(model.getVocabSize())
                                 + " (omit it to use the checkpoint's)");
    }
    return model;
}

InferenceModel load_for_inference(const std::string& checkpoint_path,
                                  const std::string& corpus_path,
                                  std::optional<int> requested_vocab) {
    GPTModel model = load_checkpoint(checkpoint_path, requested_vocab);
    BPETokenizer tokenizer(model.getVocabSize());
    load_tokenizer_for_inference(corpus_path, model.getVocabSize(), tokenizer);
    require_matching_vocab(tokenizer, model);
    return {std::move(model), std::move(tokenizer)};
}

}  // namespace cli
