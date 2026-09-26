// grad prepare: pre-tokenize a corpus once. Trains (or loads) the
// tokenizer, encodes the whole text, and writes 95/5 train/val token
// files. Training then memory-maps those files instead of re-encoding the
// corpus on every run.

#include "cli/args.h"
#include "commands.h"
#include "common.h"

#include "grad/data/token_file.h"
#include "grad/utils/metrics.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace grad::cli {

namespace {

// v1 trains on a prefix sample cut at a word boundary, exactly as before
// v2 existed, so re-running prepare --tokenizer v1 reproduces its cache.
std::unique_ptr<Tokenizer> v1_tokenizer(const std::string& corpus_path, int vocab_size,
                                        const std::string& text) {
    if (text.size() > kTokenizerSampleBytes) {
        size_t cut = text.rfind(' ', kTokenizerSampleBytes);
        if (cut == std::string::npos) cut = kTokenizerSampleBytes;
        std::cout << "Corpus is " << (text.size() >> 20) << "MB; training tokenizer on a "
                  << (cut >> 20) << "MB sample" << std::endl;
        return load_or_train_tokenizer(corpus_path, vocab_size, TokenizerKind::BpeV1,
                                       std::string_view(text).substr(0, cut));
    }
    return load_or_train_tokenizer(corpus_path, vocab_size, TokenizerKind::BpeV1, text);
}

std::unique_ptr<Tokenizer> v2_tokenizer(const std::string& corpus_path, int vocab_size,
                                        const std::string& text) {
    const std::string_view sample = tokenizer_training_text(text, TokenizerKind::ByteBpe);
    if (sample.size() < text.size()) {
        std::cout << "Corpus is " << (text.size() >> 20) << "MB; a new tokenizer learns from its "
                  << "first " << (sample.size() >> 20) << "MB" << std::endl;
    }
    return load_or_train_tokenizer(corpus_path, vocab_size, TokenizerKind::ByteBpe, sample);
}

void prepare(const std::string& corpus_path, int vocab_size, TokenizerKind kind) {
    std::cout << "\ngrad.cpp Prepare (tokenizer " << tokenizer_kind_flag(kind) << ")\n"
              << std::endl;

    utils::print_section("Tokenizing corpus");
    std::string text = read_text_file(corpus_path);
    const std::unique_ptr<Tokenizer> tokenizer = kind == TokenizerKind::BpeV1
                                                     ? v1_tokenizer(corpus_path, vocab_size, text)
                                                     : v2_tokenizer(corpus_path, vocab_size, text);

    std::cout << "Encoding text..." << std::flush;
    const Stopwatch timer;
    const std::vector<int> tokens = tokenizer->encode(text);
    const long long ms = timer.ms();
    std::cout << " " << tokens.size() << " tokens (" << ms << "ms, " << std::fixed
              << std::setprecision(1)
              << static_cast<double>(text.size()) / 1e3 / static_cast<double>(std::max(ms, 1LL))
              << " MB/s, "
              << static_cast<double>(text.size())
                     / static_cast<double>(std::max<size_t>(tokens.size(), 1))
              << " bytes/token)" << std::defaultfloat << std::endl;

    // Release the raw text before writing; the token ids are all that
    // remain live, and the split writes directly from ranges of them.
    text.clear();
    text.shrink_to_fit();

    const size_t split = tokens.size() * 95 / 100;
    const int vocab = tokenizer->vocab_size();
    const std::string train_bin = token_bin_path(corpus_path, vocab_size, "train", kind);
    const std::string val_bin = token_bin_path(corpus_path, vocab_size, "val", kind);
    tokenfile::write(train_bin, tokens.data(), split, vocab);
    tokenfile::write(val_bin, tokens.data() + split, tokens.size() - split, vocab);

    const TokenizerKind other =
        kind == TokenizerKind::BpeV1 ? TokenizerKind::ByteBpe : TokenizerKind::BpeV1;
    const bool both = is_prepared(corpus_path, vocab_size, other)
                      || std::ifstream(tokenizer_path(corpus_path, vocab_size, other)).good();
    std::cout << "\nWrote " << train_bin << " (" << split << " tokens)"
              << "\nWrote " << val_bin << " (" << (tokens.size() - split) << " tokens)"
              << "\n\nTrain with: ./build/grad train " << corpus_path
              << (both ? std::string(" --tokenizer ") + tokenizer_kind_flag(kind) : "")
              << std::endl;
}

}  // namespace

int run_prepare(const Invocation& invocation) {
    std::string corpus;
    int vocab = 5000;
    std::optional<std::string> tokenizer_flag;

    Command cmd(invocation.usage_name(), std::string(invocation.summary));
    cmd.describe(
        "Trains a BPE tokenizer on the corpus (on its first 32MB for larger corpora) and "
        "writes a 95/5 train/val split of token files that train, eval, and generate "
        "memory-map instead of re-encoding the text. v2, the default, is a lossless "
        "byte-level BPE and writes <corpus>.bytebpe_<vocab>.tok and "
        "<corpus>.v2.<vocab>.{train,val}.bin; v1 writes the original whitespace BPE's "
        "<corpus>.tokenizer_<vocab>.cache and <corpus>.<vocab>.{train,val}.bin. An "
        "existing tokenizer file is reused.");
    cmd.positional("corpus", corpus, "plain-text corpus");
    cmd.optional("vocab", vocab, "BPE vocabulary size").at_least(1);
    add_tokenizer_option(cmd, tokenizer_flag,
                         "v2 (byte-level, lossless; the default) or v1 (the original BPE)");
    if (cmd.parse(invocation.args) == ParseResult::HelpShown) return 0;

    prepare(corpus, vocab, parse_tokenizer_flag(tokenizer_flag).value_or(TokenizerKind::ByteBpe));
    return 0;
}

}  // namespace grad::cli
