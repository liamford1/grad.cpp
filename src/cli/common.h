#pragma once

// Helpers shared by more than one grad command: corpus-derived file names,
// tokenizer loading, prompt selection, and checkpoint loading for
// inference.

#include "cli/args.h"
#include "grad/data/dataset.h"
#include "grad/tokenizer/bpe_tokenizer.h"
#include "grad/transformer/gpt_model.h"

#include <chrono>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace grad::cli {

inline constexpr const char* kDefaultCorpus = "data/shakespeare.txt";

// BPE merge learning scans every unique word once per merge, so its cost
// grows with corpus size for no statistical benefit: token frequencies
// converge long before 32MB. `prepare` trains on a prefix sample this size.
inline constexpr size_t kTokenizerSampleBytes = 32ull * 1024 * 1024;

[[nodiscard]] bool is_default_corpus(const std::string& corpus_path);

// The default corpus keeps its historical cache name so existing caches
// and checkpoints stay valid; other corpora get corpus-derived names.
[[nodiscard]] std::string tokenizer_cache_prefix(const std::string& corpus_path);
// <prefix>_<vocab>.cache, where load_tokenizer caches a trained tokenizer.
[[nodiscard]] std::string tokenizer_cache_path(const std::string& corpus_path, int vocab_size);
// <corpus>.<vocab>.<split>.bin, written by `prepare`.
[[nodiscard]] std::string token_bin_path(const std::string& corpus_path, int vocab_size,
                                         const std::string& split);

// Milliseconds since construction, for the progress lines.
class Stopwatch {
public:
    [[nodiscard]] long long ms() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - start_)
            .count();
    }

private:
    std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
};

// Reads a whole file, reporting its size and read time.
[[nodiscard]] std::string read_text_file(const std::string& path);

// Loads <cache_prefix>_<vocab>.cache if it exists; otherwise trains the
// tokenizer on text and writes that cache.
void load_tokenizer(const std::string& text, const std::string& cache_prefix, int vocab_size,
                    BPETokenizer& tokenizer);

// A generation prompt as both display text and token ids. Held-out prompts
// keep the ids they were drawn with: the tokenizer drops whitespace, so
// decode-then-encode would not give back the same tokens.
struct Prompt {
    std::string text;
    std::vector<int> tokens;
};

[[nodiscard]] Prompt prompt_from_text(const BPETokenizer& tokenizer, const std::string& text);

// Prompts for end-of-run samples and for `generate` without a prompt.
// Shakespeare keeps its speaker tags, which the model learns to continue in
// character. Any other corpus has no known structure, so the prompts are the
// opening tokens of `count` held-out windows spread evenly across the val
// split: text the model has not trained on, in the corpus's own style.
[[nodiscard]] std::vector<Prompt> sample_prompts(const std::string& corpus_path,
                                                 const BPETokenizer& tokenizer,
                                                 const Dataset& val, size_t count = 3);

// Decoding settings for generate and chat, as TextGen takes them.
struct SamplingOptions {
    int max_tokens;
    float temperature = 0.8f;
    int top_k = 0;       // 0 = no top-k cut
    float top_p = 1.0f;  // 1 = no nucleus cut
    float repetition_penalty = 1.2f;
    bool greedy = false;
};

// Declares --max-tokens, --temperature, --top-k, --top-p,
// --repetition-penalty and --greedy on cmd, bound to options; the current
// values are the defaults.
void add_sampling_options(Command& cmd, SamplingOptions& options, const std::string& greedy_help);

// A checkpoint and the tokenizer it was trained with, for generate/chat.
struct InferenceModel {
    GPTModel model;
    BPETokenizer tokenizer;
};

// Loads a checkpoint and, for generate/chat, its corpus's tokenizer. The
// vocab is the one the user gave, else the checkpoint's own; a mismatch
// with the checkpoint is an error before any tokenizer is loaded, since a
// missing cache for a wrong vocab would otherwise start BPE training on
// the corpus.
[[nodiscard]] GPTModel load_checkpoint(const std::string& checkpoint_path,
                                       std::optional<int> requested_vocab);
[[nodiscard]] InferenceModel load_for_inference(const std::string& checkpoint_path,
                                                const std::string& corpus_path,
                                                std::optional<int> requested_vocab);

}  // namespace grad::cli
