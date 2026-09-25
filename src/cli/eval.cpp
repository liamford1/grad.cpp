// grad eval: score a checkpoint on the held-out split and on an
// equal-sized random sample of training windows, both in non-overlapping
// windows. The trainer's in-loop validation reads only a fixed subset of
// the split, which is fine for tracking a run but not for reporting one;
// this is the number to report. max_batches = 0 walks the whole val split;
// otherwise both splits are sampled uniformly, so a capped run still covers
// the split rather than its first few stories. The train-side figure
// separates generalization from distribution shift between the two splits.

#include "cli/args.h"
#include "commands.h"
#include "common.h"

#include "grad/data/dataloader.h"
#include "grad/data/token_file.h"
#include "grad/training/trainer.h"
#include "grad/utils/metrics.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace cli {

namespace {

void evaluate(const std::string& checkpoint_path, const std::string& corpus_path,
              std::optional<int> requested_vocab, int seq_length, int max_batches) {
    std::cout << "\ngrad.cpp Evaluation\n" << std::endl;

    GPTModel model = load_checkpoint(checkpoint_path, requested_vocab);
    const int vocab_size = model.getVocabSize();

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

    const auto report = [](const char* split, double loss, long tokens, double seconds) {
        std::cout << std::left << std::setw(8) << split << std::right << std::fixed
                  << " loss " << std::setprecision(4) << loss
                  << "  perplexity " << std::setprecision(3) << std::exp(loss)
                  << "  (" << tokens << " tokens, " << std::setprecision(0) << seconds
                  << "s)" << std::defaultfloat << std::endl;
    };
    const auto timed = [&](DataLoader& loader) {
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
    report("train", train_loss, static_cast<long>(val_batches) * kBatchSize * seq_length, train_s);
}

}  // namespace

int run_eval(const Invocation& invocation) {
    std::string checkpoint;
    std::string corpus;
    std::optional<int> vocab;
    int seq = 256;
    int max_batches = 0;

    Command cmd(invocation.usage_name(), std::string(invocation.summary));
    cmd.describe("Scores the val split and an equal-sized uniform sample of training windows "
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
    if (cmd.parse(invocation.args) == ParseResult::HelpShown) return 0;

    evaluate(checkpoint, corpus, vocab, seq, max_batches);
    return 0;
}

}  // namespace cli
