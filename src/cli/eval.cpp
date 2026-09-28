// grad eval: score a checkpoint on the held-out split and on an
// equal-sized random sample of training windows, both in non-overlapping
// windows. The trainer's in-loop validation reads only a fixed subset of
// the split, which is fine for tracking a run but not for reporting one;
// this is the number to report. max_batches = 0 walks the whole val split;
// otherwise both splits are sampled uniformly, so a capped run still covers
// the split rather than its first few stories. The train-side figure
// separates generalization from distribution shift between the two splits.
//
// Bits per byte makes runs on different tokenizers comparable, which
// per-token loss is not: loss / ln 2 / (corpus bytes per token), with the
// ratio taken over the whole prepared corpus (train + val tokens against
// the corpus file's size). v2 spends ~20% more tokens than v1 on the same
// text, so the same bits per byte shows up as a ~20% lower v1 loss. The
// ratio is a corpus average, so a window's own byte count may differ; over
// a large uniform sample the difference averages out.

#include "cli/args.h"
#include "commands.h"
#include "common.h"

#include "grad/data/dataloader.h"
#include "grad/data/token_file.h"
#include "grad/training/trainer.h"
#include "grad/utils/metrics.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <numbers>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace grad::cli {

namespace {

void evaluate(const std::string& checkpoint_path, const std::string& corpus_path,
              std::optional<int> requested_vocab, std::optional<TokenizerKind> requested_tokenizer,
              int seq_length, int max_batches) {
    std::cout << "\ngrad.cpp Evaluation\n" << std::endl;

    GPTModel model = load_checkpoint(checkpoint_path, requested_vocab);
    const int vocab_size = model.getVocabSize();
    const TokenizerKind kind =
        choose_tokenizer(corpus_path, vocab_size, requested_tokenizer, &model);

    const std::string train_bin = token_bin_path(corpus_path, vocab_size, "train", kind);
    const std::string val_bin = token_bin_path(corpus_path, vocab_size, "val", kind);
    if (!tokenfile::exists(train_bin) || !tokenfile::exists(val_bin)) {
        throw std::runtime_error("eval needs pre-tokenized files; run: ./build/grad prepare "
                                 + corpus_path + " " + std::to_string(vocab_size) + " --tokenizer "
                                 + tokenizer_kind_flag(kind));
    }
    // The token files were written by the tokenizer beside them; checking
    // that one against the checkpoint checks the files.
    if (std::ifstream(tokenizer_path(corpus_path, vocab_size, kind)).good()) {
        const std::unique_ptr<Tokenizer> tokenizer =
            load_existing_tokenizer(corpus_path, vocab_size, kind);
        check_tokenizer_matches(model, *tokenizer, checkpoint_path);
    } else {
        std::cerr << "Warning: no tokenizer file beside " << val_bin
                  << "; cannot check the token files against the checkpoint" << std::endl;
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
    const int val_batches =
        max_batches > 0 ? std::min<int>(max_batches, static_cast<int>(val_loader.num_batches()))
                        : static_cast<int>(val_loader.num_batches());
    DataLoader train_loader(train, kBatchSize, /*shuffle=*/true, kSeed + 1);

    // Corpus bytes per token, for bits per byte; unknown when the corpus
    // text is not present beside its prepared token files.
    std::error_code size_error;
    const auto corpus_bytes = std::filesystem::file_size(corpus_path, size_error);
    const double bytes_per_token =
        size_error ? 0.0
                   : static_cast<double>(corpus_bytes)
                         / static_cast<double>(train->tokenCount() + val->tokenCount());
    if (bytes_per_token > 0.0) {
        std::cout << "Corpus: " << std::fixed << std::setprecision(3) << bytes_per_token
                  << " bytes/token" << std::defaultfloat << std::endl;
    } else {
        std::cerr << "Warning: " << corpus_path
                  << " not found; bits per byte needs its size and is not reported" << std::endl;
    }

    const auto report = [bytes_per_token](const char* split, double loss, long tokens,
                                          double seconds) {
        std::cout << std::left << std::setw(8) << split << std::right << std::fixed << " loss "
                  << std::setprecision(4) << loss << "  perplexity " << std::setprecision(3)
                  << std::exp(loss);
        if (bytes_per_token > 0.0) {
            std::cout << "  bits/byte " << std::setprecision(4)
                      << loss / std::numbers::ln2 / bytes_per_token;
        }
        std::cout << "  (" << tokens << " tokens, " << std::setprecision(0) << seconds << "s)"
                  << std::defaultfloat << std::endl;
    };
    const auto timed = [&](DataLoader& loader) {
        const auto start = std::chrono::steady_clock::now();
        const double loss = training::mean_loss(model, loader, val_batches);
        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
        return std::make_pair(loss, elapsed.count());
    };

    utils::print_section("Scoring");
    const long tokens = static_cast<long>(std::min<size_t>(
                            static_cast<size_t>(val_batches) * kBatchSize, val->size()))
                        * seq_length;
    const auto [val_loss, val_s] = timed(val_loader);
    report("val", val_loss, tokens, val_s);
    const auto [train_loss, train_s] = timed(train_loader);
    report("train", train_loss, static_cast<long>(val_batches) * kBatchSize * seq_length, train_s);
}

}  // namespace

int run_eval(const Invocation& invocation) {
    std::string checkpoint;
    std::string corpus;
    std::optional<int> vocab;
    int seq = 256;
    int max_batches = 0;
    std::optional<std::string> tokenizer_flag;
    std::optional<std::string> device;

    Command cmd(invocation.usage_name(), std::string(invocation.summary));
    cmd.describe(
        "Scores the val split and an equal-sized uniform sample of training windows "
        "in non-overlapping windows with a fixed seed, so every checkpoint sees the "
        "same windows. Needs the corpus's prepared token files (see prepare).");
    cmd.positional("ckpt", checkpoint, "checkpoint to score");
    cmd.positional("corpus", corpus, "the corpus the checkpoint was trained on");
    cmd.optional("vocab", vocab, "vocab size; must match the checkpoint's")
        .at_least(1)
        .default_text("the checkpoint's");
    cmd.optional("seq", seq, "window length in tokens").at_least(1);
    cmd.optional("max_batches", max_batches,
                 "batches of 8 windows per split; 0 scores the whole val split")
        .at_least(0)
        .named("--batches");
    add_tokenizer_option(cmd, tokenizer_flag, kInferenceTokenizerHelp);
    add_device_option(cmd, device);
    if (cmd.parse(invocation.args) == ParseResult::HelpShown) return 0;

    const std::optional<TokenizerKind> kind = parse_tokenizer_flag(tokenizer_flag);
    apply_device(device);
    evaluate(checkpoint, corpus, vocab, kind, seq, max_batches);
    return 0;
}

}  // namespace grad::cli
