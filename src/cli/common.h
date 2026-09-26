#pragma once

// Helpers shared by more than one grad command: corpus-derived file names,
// choosing and loading the tokenizer (v1 or v2), prompt selection, and
// checkpoint loading for inference.

#include "cli/args.h"
#include "grad/data/dataset.h"
#include "grad/tokenizer/tokenizer.h"
#include "grad/transformer/gpt_model.h"

#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace grad::cli {

inline constexpr const char* kDefaultCorpus = "data/shakespeare.txt";

// Merge learning gains nothing statistically past a few tens of MB (token
// frequencies converge long before), so larger corpora learn their merges
// from a prefix this size. v1's cost also grows with it, since it rescans
// every word per merge.
inline constexpr size_t kTokenizerSampleBytes = 32ull * 1024 * 1024;

[[nodiscard]] bool is_default_corpus(const std::string& corpus_path);

// v1 artifacts keep their historical names, so existing caches, token files
// and checkpoints stay valid: the default corpus's cache is
// tokenizer_<vocab>.cache in the working directory, any other corpus's is
// <corpus>.tokenizer_<vocab>.cache, and token files are
// <corpus>.<vocab>.<split>.bin. v2 artifacts use names v1 never does:
// <corpus>.bytebpe_<vocab>.tok and <corpus>.v2.<vocab>.<split>.bin.
[[nodiscard]] std::string tokenizer_cache_prefix(const std::string& corpus_path);
[[nodiscard]] std::string tokenizer_path(const std::string& corpus_path, int vocab_size,
                                         TokenizerKind kind);
[[nodiscard]] std::string token_bin_path(const std::string& corpus_path, int vocab_size,
                                         const std::string& split, TokenizerKind kind);
// Both of a corpus's token files (train and val) exist for this kind.
[[nodiscard]] bool is_prepared(const std::string& corpus_path, int vocab_size, TokenizerKind kind);

// --tokenizer help for the commands that load a checkpoint.
inline constexpr const char* kInferenceTokenizerHelp =
    "v1 or v2; by default the one the checkpoint records (v1 for checkpoints that record "
    "none, which all predate v2)";

// Declares --tokenizer v1|v2 on cmd.
void add_tokenizer_option(Command& cmd, std::optional<std::string>& flag, const std::string& help);
// The kind a --tokenizer value names; throws UsageError for anything but
// "v1" and "v2".
[[nodiscard]] std::optional<TokenizerKind> parse_tokenizer_flag(
    const std::optional<std::string>& flag);

// Which tokenizer a command uses for corpus + vocab:
//  1. the kind recorded in the checkpoint (an explicit --tokenizer that
//     contradicts it is an error);
//  2. an explicit --tokenizer;
//  3. v1 for a checkpoint that records none, since every such checkpoint
//     was written before v2 existed;
//  4. the kind whose tokenizer or token files exist for the corpus: an
//     error if both do, v2 (for a new tokenizer) if neither does.
[[nodiscard]] TokenizerKind choose_tokenizer(const std::string& corpus_path, int vocab_size,
                                             std::optional<TokenizerKind> requested,
                                             const GPTModel* checkpoint);

// The text a new tokenizer of this kind learns from. v2 uses the first
// kTokenizerSampleBytes of a larger corpus, cut after a newline, whichever
// command trains it; v1 keeps its historical behavior (the full text here;
// prepare cuts its own sample).
[[nodiscard]] std::string_view tokenizer_training_text(std::string_view text, TokenizerKind kind);

// Loads the corpus's tokenizer of this kind, or trains it on
// training_text and saves it there first when the file does not exist.
[[nodiscard]] std::unique_ptr<Tokenizer> load_or_train_tokenizer(const std::string& corpus_path,
                                                                 int vocab_size, TokenizerKind kind,
                                                                 std::string_view training_text);
// Loads the corpus's tokenizer file of this kind, which must exist.
[[nodiscard]] std::unique_ptr<Tokenizer> load_existing_tokenizer(const std::string& corpus_path,
                                                                 int vocab_size,
                                                                 TokenizerKind kind);

// Refuses a tokenizer whose fingerprint differs from the one the
// checkpoint records, and warns when the checkpoint records none.
void check_tokenizer_matches(const GPTModel& model, const Tokenizer& tokenizer,
                             const std::string& checkpoint_path);

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

// A generation prompt as both display text and token ids. Held-out prompts
// keep the ids they were drawn with: v1 drops whitespace, and a window can
// start inside a word, so decode-then-encode need not give back the same
// tokens.
struct Prompt {
    std::string text;
    std::vector<int> tokens;
};

[[nodiscard]] Prompt prompt_from_text(const Tokenizer& tokenizer, const std::string& text);

// Prompts for end-of-run samples and for `generate` without a prompt.
// Shakespeare keeps its speaker tags, which the model learns to continue in
// character. Any other corpus has no known structure, so the prompts are the
// opening tokens of `count` held-out windows spread evenly across the val
// split: text the model has not trained on, in the corpus's own style. With
// v2, whose "<|endoftext|>" is a token, each prompt starts at the next
// document boundary after its window when there is one nearby, so it opens a
// document (a story, for TinyStories) rather than cutting into one.
[[nodiscard]] std::vector<Prompt> sample_prompts(const std::string& corpus_path,
                                                 const Tokenizer& tokenizer, const Dataset& val,
                                                 size_t count = 3);

// Decoding settings for generate and chat, as TextGen takes them.
struct SamplingOptions {
    int max_tokens = 0;
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
    std::unique_ptr<Tokenizer> tokenizer;
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
                                                std::optional<int> requested_vocab,
                                                std::optional<TokenizerKind> requested_tokenizer);

}  // namespace grad::cli
