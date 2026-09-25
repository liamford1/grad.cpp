#include "common.h"

#include "grad/data/token_file.h"
#include "grad/tokenizer/bpe_v1.h"
#include "grad/tokenizer/byte_bpe.h"
#include "grad/utils/metrics.h"

#include <sys/stat.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>

namespace grad::cli {

namespace {

bool file_exists(const std::string& path) {
    return std::ifstream(path).good();
}

std::string vocab_args(const std::string& corpus_path, int vocab_size, TokenizerKind kind) {
    return corpus_path + " " + std::to_string(vocab_size) + " --tokenizer "
           + tokenizer_kind_flag(kind);
}

// Either the tokenizer file or both token files exist.
bool has_artifacts(const std::string& corpus_path, int vocab_size, TokenizerKind kind) {
    return file_exists(tokenizer_path(corpus_path, vocab_size, kind))
           || is_prepared(corpus_path, vocab_size, kind);
}

// Inference-time tokenizer loading: the tokenizer file must exist
// (train/prepare created it), so the corpus text - possibly gigabytes - is
// never read. Falls back to training from text only for small unprepared
// corpora; a large one would take long and, unlike `prepare`, v1 would
// train on the full text rather than its sample, giving a different
// tokenizer.
std::unique_ptr<Tokenizer> load_tokenizer_for_inference(const std::string& corpus_path,
                                                        int vocab_size, TokenizerKind kind) {
    if (file_exists(tokenizer_path(corpus_path, vocab_size, kind))) {
        return load_existing_tokenizer(corpus_path, vocab_size, kind);
    }
    struct stat st{};
    if (::stat(corpus_path.c_str(), &st) == 0
        && static_cast<size_t>(st.st_size) > kTokenizerSampleBytes) {
        throw std::runtime_error("No tokenizer " + tokenizer_path(corpus_path, vocab_size, kind)
                                 + " for a " + std::to_string(st.st_size >> 20)
                                 + "MB corpus; run: ./build/grad prepare "
                                 + vocab_args(corpus_path, vocab_size, kind));
    }
    const std::string text = read_text_file(corpus_path);
    return load_or_train_tokenizer(corpus_path, vocab_size, kind,
                                   tokenizer_training_text(text, kind));
}

void require_matching_vocab(const Tokenizer& tokenizer, const GPTModel& model) {
    if (tokenizer.vocab_size() != model.getVocabSize()) {
        throw std::runtime_error("tokenizer has " + std::to_string(tokenizer.vocab_size())
                                 + " tokens but the checkpoint expects "
                                 + std::to_string(model.getVocabSize())
                                 + "; the cache was built for a different vocab or corpus");
    }
}

// The first `count` tokens of the text after the first `boundary` token at
// or after window `index` (whitespace-only tokens right after the boundary
// are skipped, so the prompt starts at the text), scanning at most `limit`
// tokens; empty if there is none. Assumes non-overlapping windows (stride =
// window length), which is how every caller builds its val split, so
// consecutive windows are consecutive text.
std::vector<int> tokens_after_boundary(const Tokenizer& tokenizer, const Dataset& val, size_t index,
                                       int boundary, size_t count, size_t limit) {
    const auto whitespace = [&](int token) {
        const std::string text = tokenizer.decode(std::span<const int>(&token, 1));
        return std::all_of(text.begin(), text.end(),
                           [](char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; });
    };
    std::vector<int> out;
    bool found = false;
    size_t scanned = 0;
    for (size_t w = index; w < val.size() && scanned < limit; w++) {
        for (const int token : val.get_item(w).first) {
            scanned++;
            if (!found) {
                found = token == boundary;
            } else if (!out.empty() || !whitespace(token)) {
                out.push_back(token);
                if (out.size() == count) return out;
            }
        }
    }
    return {};
}

}  // namespace

bool is_default_corpus(const std::string& corpus_path) {
    return corpus_path == kDefaultCorpus;
}

std::string tokenizer_cache_prefix(const std::string& corpus_path) {
    if (is_default_corpus(corpus_path)) return "tokenizer";
    return corpus_path + ".tokenizer";
}

std::string tokenizer_path(const std::string& corpus_path, int vocab_size, TokenizerKind kind) {
    if (kind == TokenizerKind::ByteBpe) {
        return corpus_path + ".bytebpe_" + std::to_string(vocab_size) + ".tok";
    }
    return tokenizer_cache_prefix(corpus_path) + "_" + std::to_string(vocab_size) + ".cache";
}

std::string token_bin_path(const std::string& corpus_path, int vocab_size, const std::string& split,
                           TokenizerKind kind) {
    const std::string tag = kind == TokenizerKind::ByteBpe ? ".v2." : ".";
    return corpus_path + tag + std::to_string(vocab_size) + "." + split + ".bin";
}

bool is_prepared(const std::string& corpus_path, int vocab_size, TokenizerKind kind) {
    return tokenfile::exists(token_bin_path(corpus_path, vocab_size, "train", kind))
           && tokenfile::exists(token_bin_path(corpus_path, vocab_size, "val", kind));
}

void add_tokenizer_option(Command& cmd, std::optional<std::string>& flag, const std::string& help) {
    cmd.option("--tokenizer", flag, help).metavar("v1|v2");
}

std::optional<TokenizerKind> parse_tokenizer_flag(const std::optional<std::string>& flag) {
    if (!flag) return std::nullopt;
    if (*flag == "v1") return TokenizerKind::BpeV1;
    if (*flag == "v2") return TokenizerKind::ByteBpe;
    throw UsageError("--tokenizer: expected v1 or v2, got '" + *flag + "'");
}

TokenizerKind choose_tokenizer(const std::string& corpus_path, int vocab_size,
                               std::optional<TokenizerKind> requested, const GPTModel* checkpoint) {
    const std::optional<TokenizerFingerprint> recorded_fingerprint =
        checkpoint ? checkpoint->getTokenizerFingerprint() : std::nullopt;
    if (recorded_fingerprint) {
        const TokenizerFingerprint& recorded = *recorded_fingerprint;
        if (requested && *requested != recorded.kind) {
            throw std::runtime_error(std::string("--tokenizer ") + tokenizer_kind_flag(*requested)
                                     + " does not match the checkpoint, which was trained with "
                                     + "tokenizer " + to_string(recorded));
        }
        return recorded.kind;
    }
    if (requested) return *requested;
    if (checkpoint) return TokenizerKind::BpeV1;

    const bool v1 = has_artifacts(corpus_path, vocab_size, TokenizerKind::BpeV1);
    const bool v2 = has_artifacts(corpus_path, vocab_size, TokenizerKind::ByteBpe);
    if (v1 && v2) {
        throw std::runtime_error(
            "both a v1 tokenizer (" + tokenizer_path(corpus_path, vocab_size, TokenizerKind::BpeV1)
            + ") and a v2 tokenizer ("
            + tokenizer_path(corpus_path, vocab_size, TokenizerKind::ByteBpe) + ") exist for "
            + corpus_path + " at vocab " + std::to_string(vocab_size)
            + "; pass --tokenizer v1 or --tokenizer v2");
    }
    return v1 ? TokenizerKind::BpeV1 : TokenizerKind::ByteBpe;
}

std::string_view tokenizer_training_text(std::string_view text, TokenizerKind kind) {
    if (kind == TokenizerKind::BpeV1 || text.size() <= kTokenizerSampleBytes) return text;
    size_t cut = text.rfind('\n', kTokenizerSampleBytes - 1);
    if (cut == std::string_view::npos) cut = text.rfind(' ', kTokenizerSampleBytes - 1);
    return text.substr(0, cut == std::string_view::npos ? kTokenizerSampleBytes : cut + 1);
}

std::unique_ptr<Tokenizer> load_or_train_tokenizer(const std::string& corpus_path, int vocab_size,
                                                   TokenizerKind kind,
                                                   std::string_view training_text) {
    const std::string path = tokenizer_path(corpus_path, vocab_size, kind);
    std::unique_ptr<Tokenizer> tokenizer;
    if (kind == TokenizerKind::BpeV1) {
        // v1's messages and behavior, unchanged.
        if (file_exists(path)) {
            std::cout << "Loading tokenizer from cache..." << std::flush;
            const Stopwatch timer;
            tokenizer = std::make_unique<BpeV1>(BpeV1::load(path));
            std::cout << " done (" << timer.ms() << "ms)" << std::endl;
        } else {
            std::cout << "Training new tokenizer..." << std::flush;
            const Stopwatch timer;
            tokenizer =
                std::make_unique<BpeV1>(BpeV1::train(std::string(training_text), vocab_size));
            std::cout << " done (" << timer.ms() << "ms)" << std::endl;
            tokenizer->save(path);
            std::cout << "Cached to " << path << std::endl;
        }
    } else if (file_exists(path)) {
        std::cout << "Loading v2 tokenizer from " << path << "..." << std::flush;
        const Stopwatch timer;
        tokenizer = load_existing_tokenizer(corpus_path, vocab_size, kind);
        std::cout << " done (" << timer.ms() << "ms)" << std::endl;
    } else {
        std::cout << "Training new v2 tokenizer (byte-level BPE) on "
                  << (training_text.size() >> 10) << "KB" << std::endl;
        const Stopwatch timer;
        tokenizer = std::make_unique<ByteBpe>(ByteBpe::train(
            training_text, vocab_size, {std::string(ByteBpe::kEndOfText)}, {.progress = true}));
        std::cout << "Trained in " << timer.ms() << "ms" << std::endl;
        tokenizer->save(path);
        std::cout << "Saved " << path << std::endl;
    }
    std::cout << "Vocab size: " << tokenizer->vocab_size() << std::endl;
    return tokenizer;
}

std::unique_ptr<Tokenizer> load_existing_tokenizer(const std::string& corpus_path, int vocab_size,
                                                   TokenizerKind kind) {
    const std::string path = tokenizer_path(corpus_path, vocab_size, kind);
    if (!file_exists(path)) {
        throw std::runtime_error("No " + std::string(tokenizer_kind_flag(kind)) + " tokenizer "
                                 + path + "; run: ./build/grad prepare "
                                 + vocab_args(corpus_path, vocab_size, kind));
    }
    std::unique_ptr<Tokenizer> tokenizer = load_tokenizer(path);
    if (tokenizer->kind() != kind) {
        throw std::runtime_error(path + " holds a " + tokenizer_kind_flag(tokenizer->kind())
                                 + " tokenizer, not " + tokenizer_kind_flag(kind));
    }
    std::cout << "Tokenizer: " << path << " (" << tokenizer_kind_flag(kind) << ", vocab "
              << tokenizer->vocab_size() << ")" << std::endl;
    return tokenizer;
}

void check_tokenizer_matches(const GPTModel& model, const Tokenizer& tokenizer,
                             const std::string& checkpoint_path) {
    const std::optional<TokenizerFingerprint>& recorded = model.getTokenizerFingerprint();
    if (!recorded) {
        std::cerr << "Warning: " << checkpoint_path
                  << " records no tokenizer fingerprint (it predates tokenizer v2); using the "
                  << tokenizer_kind_flag(tokenizer.kind()) << " tokenizer unverified" << std::endl;
        return;
    }
    if (*recorded != tokenizer.identity()) {
        throw std::runtime_error("tokenizer " + to_string(tokenizer.identity()) + " is not the one "
                                 + checkpoint_path + " was trained with (" + to_string(*recorded)
                                 + "); its token ids would mean different text");
    }
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

Prompt prompt_from_text(const Tokenizer& tokenizer, const std::string& text) {
    return {text, tokenizer.encode(text)};
}

std::vector<Prompt> sample_prompts(const std::string& corpus_path, const Tokenizer& tokenizer,
                                   const Dataset& val, size_t count) {
    if (is_default_corpus(corpus_path)) {
        return {prompt_from_text(tokenizer, "ROMEO:\n"), prompt_from_text(tokenizer, "JULIET:\n"),
                prompt_from_text(tokenizer, "First Citizen:\n")};
    }
    constexpr size_t kPromptTokens = 6;
    constexpr size_t kBoundaryScanTokens = 4096;
    // v1's "<eos>" never occurs in encoded text, so only v2 looks for one.
    const std::optional<int> boundary =
        tokenizer.kind() == TokenizerKind::ByteBpe ? tokenizer.eos_id() : std::nullopt;
    std::vector<Prompt> prompts;
    const size_t windows = val.size();
    for (size_t k = 0; k < count && windows > 0; k++) {
        // Midpoints of `count` equal slices of the split.
        const size_t index = windows * (2 * k + 1) / (2 * count);
        std::vector<int> ids;
        if (boundary) {
            ids = tokens_after_boundary(tokenizer, val, index, *boundary, kPromptTokens,
                                        kBoundaryScanTokens);
        }
        if (ids.empty()) {
            ids = val.get_item(index).first;
            ids.resize(std::min(ids.size(), kPromptTokens));
        }
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
                                  std::optional<int> requested_vocab,
                                  std::optional<TokenizerKind> requested_tokenizer) {
    GPTModel model = load_checkpoint(checkpoint_path, requested_vocab);
    const TokenizerKind kind =
        choose_tokenizer(corpus_path, model.getVocabSize(), requested_tokenizer, &model);
    std::unique_ptr<Tokenizer> tokenizer =
        load_tokenizer_for_inference(corpus_path, model.getVocabSize(), kind);
    require_matching_vocab(*tokenizer, model);
    check_tokenizer_matches(model, *tokenizer, checkpoint_path);
    return {std::move(model), std::move(tokenizer)};
}

}  // namespace grad::cli
