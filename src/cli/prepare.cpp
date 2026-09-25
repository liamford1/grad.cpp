// grad prepare: pre-tokenize a corpus once. Trains (or loads) the BPE
// tokenizer, encodes the whole text, and writes 95/5 train/val token
// files. Training then memory-maps those files instead of re-encoding the
// corpus on every run.

#include "cli/args.h"
#include "commands.h"
#include "common.h"

#include "grad/data/token_file.h"
#include "grad/utils/metrics.h"

#include <iostream>
#include <string>
#include <vector>

namespace grad::cli {

namespace {

void prepare(const std::string& corpus_path, int vocab_size) {
    std::cout << "\ngrad.cpp Prepare\n" << std::endl;

    utils::print_section("Tokenizing corpus");
    std::string text = read_text_file(corpus_path);
    BPETokenizer tokenizer(vocab_size);

    // Train the tokenizer on a prefix sample (cut at a word boundary; see
    // kTokenizerSampleBytes), then encode the full corpus with it.
    if (text.size() > kTokenizerSampleBytes) {
        size_t cut = text.rfind(' ', kTokenizerSampleBytes);
        if (cut == std::string::npos) cut = kTokenizerSampleBytes;
        std::cout << "Corpus is " << (text.size() >> 20) << "MB; training tokenizer on a "
                  << (cut >> 20) << "MB sample" << std::endl;
        const std::string sample = text.substr(0, cut);
        load_tokenizer(sample, tokenizer_cache_prefix(corpus_path), vocab_size, tokenizer);
    } else {
        load_tokenizer(text, tokenizer_cache_prefix(corpus_path), vocab_size, tokenizer);
    }

    std::cout << "Encoding text..." << std::flush;
    const Stopwatch timer;
    const std::vector<int> tokens = tokenizer.encode(text);
    std::cout << " " << tokens.size() << " tokens (" << timer.ms() << "ms)" << std::endl;

    // Release the raw text before writing; the token ids are all that
    // remain live, and the split writes directly from ranges of them.
    text.clear();
    text.shrink_to_fit();

    const size_t split = tokens.size() * 95 / 100;
    const int vocab = tokenizer.getCurrentVocabSize();
    const std::string train_bin = token_bin_path(corpus_path, vocab_size, "train");
    const std::string val_bin = token_bin_path(corpus_path, vocab_size, "val");
    tokenfile::write(train_bin, tokens.data(), split, vocab);
    tokenfile::write(val_bin, tokens.data() + split, tokens.size() - split, vocab);

    std::cout << "\nWrote " << train_bin << " (" << split << " tokens)"
              << "\nWrote " << val_bin << " (" << (tokens.size() - split) << " tokens)"
              << "\n\nTrain with: ./build/grad train " << corpus_path << std::endl;
}

}  // namespace

int run_prepare(const Invocation& invocation) {
    std::string corpus;
    int vocab = 5000;

    Command cmd(invocation.usage_name(), std::string(invocation.summary));
    cmd.describe(
        "Trains a BPE tokenizer on the corpus (on its first 32MB for larger "
        "corpora) and writes <corpus>.<vocab>.train.bin and .val.bin, a 95/5 split "
        "that train, eval, and generate memory-map instead of re-encoding the text.");
    cmd.positional("corpus", corpus, "plain-text corpus");
    cmd.optional("vocab", vocab, "BPE vocabulary size").at_least(1);
    if (cmd.parse(invocation.args) == ParseResult::HelpShown) return 0;

    prepare(corpus, vocab);
    return 0;
}

}  // namespace grad::cli
