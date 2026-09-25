// grad train / grad train-fast: train a preset on a corpus, from scratch,
// from a resume pair, or warm-started from a checkpoint.

#include "cli/args.h"
#include "commands.h"
#include "common.h"
#include "presets.h"

#include "grad/data/dataloader.h"
#include "grad/data/dataset.h"
#include "grad/data/token_file.h"
#include "grad/training/trainer.h"
#include "grad/transformer/activations.h"
#include "grad/transformer/tensor.h"
#include "grad/transformer/text_gen.h"
#include "grad/utils/metrics.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace grad::cli {

namespace {

// Checkpoint prefix from the corpus filename: data/tinystories.txt ->
// "tinystories" (data/shakespeare.txt keeps its historical "shakespeare").
// train_supervised.sh derives the same prefix; keep the two in step.
std::string checkpoint_stem(const std::string& corpus_path) {
    const size_t slash = corpus_path.find_last_of('/');
    const std::string base =
        (slash == std::string::npos) ? corpus_path : corpus_path.substr(slash + 1);
    const size_t dot = base.find_last_of('.');
    return (dot == std::string::npos) ? base : base.substr(0, dot);
}

std::string checkpoint_prefix(const std::string& corpus_path, const Preset& preset) {
    return checkpoint_stem(corpus_path) + (preset.modern ? "_modern" : "")
         + (preset.is_smoke_test() ? "_fast" : "");
}

struct TrainingData {
    std::shared_ptr<Dataset> train;
    std::shared_ptr<Dataset> val;
};

// Memory-maps the corpus's prepared token files, whose tokenizer cache
// must also exist. The corpus text is never loaded and nothing is
// re-encoded, so startup cost and memory use are independent of corpus
// size.
TrainingData map_prepared(const std::string& corpus_path, int vocab_size, int seq_length,
                          BPETokenizer& tokenizer) {
    utils::print_section("Loading Data (pre-tokenized)");
    const std::string cache_file = tokenizer_cache_path(corpus_path, vocab_size);
    if (!std::ifstream(cache_file).good()) {
        throw std::runtime_error("Found token files but no tokenizer cache (" + cache_file
                                 + "); run: ./build/grad prepare " + corpus_path);
    }
    tokenizer.load(cache_file);

    const std::string train_bin = token_bin_path(corpus_path, vocab_size, "train");
    auto train = std::make_shared<MappedTokenDataset>(train_bin, seq_length);
    auto val = std::make_shared<MappedTokenDataset>(token_bin_path(corpus_path, vocab_size, "val"),
                                                    seq_length, seq_length);
    if (train->vocabSize() != tokenizer.getCurrentVocabSize()) {
        throw std::runtime_error("Token file vocab does not match tokenizer cache; re-run prepare");
    }
    std::cout << "Mapped " << train->tokenCount() << " train / " << val->tokenCount()
              << " val tokens from " << train_bin << std::endl;
    return {train, val};
}

// Reads and encodes the corpus now. Fine for small corpora; for anything
// large, run `prepare` first.
TrainingData encode_in_memory(const std::string& corpus_path, int vocab_size, int seq_length,
                              BPETokenizer& tokenizer) {
    utils::print_section("Loading Data");
    const std::string text = read_text_file(corpus_path);
    load_tokenizer(text, tokenizer_cache_prefix(corpus_path), vocab_size, tokenizer);

    std::cout << "Encoding text..." << std::flush;
    const Stopwatch timer;
    const std::vector<int> tokens = tokenizer.encode(text);
    std::cout << " " << tokens.size() << " tokens (" << timer.ms() << "ms)" << std::endl;

    // Hold out the last 5% of the corpus for validation. The split is
    // contiguous, so no training window ever overlaps validation text -
    // val perplexity measures generalization, not memorization.
    const std::span<const int> all(tokens);
    const size_t split = all.size() * 95 / 100;
    const std::vector<int> train_tokens(all.first(split).begin(), all.first(split).end());
    const std::vector<int> val_tokens(all.subspan(split).begin(), all.subspan(split).end());
    // Non-overlapping val windows: evaluation covers the whole held-out
    // slice once, deterministically.
    return {std::make_shared<TextDataset>(train_tokens, seq_length),
            std::make_shared<TextDataset>(val_tokens, seq_length, seq_length)};
}

training::TrainingConfig make_config(const Preset& preset, int vocab_size, std::string prefix) {
    training::TrainingConfig config;
    config.vocab_size = vocab_size;
    config.d_model = preset.d_model;
    config.num_layers = preset.num_layers;
    config.num_heads = preset.num_heads;
    config.max_len = preset.max_len;
    config.seq_length = preset.seq_length;
    config.batch_size = preset.batch_size;
    config.grad_accum = preset.grad_accum;
    config.learning_rate = preset.learning_rate;
    config.dropout = preset.dropout;
    config.warmup_steps = preset.warmup_steps;
    config.num_steps = preset.num_steps;
    config.checkpoint_interval = preset.checkpoint_interval;
    config.checkpoint_prefix = std::move(prefix);
    config.eval_interval = preset.eval_interval;
    config.max_eval_batches = preset.max_eval_batches;
    return config;
}

void generate_samples(GPTModel& model, const BPETokenizer& tokenizer,
                      const std::vector<Prompt>& prompts) {
    utils::print_section("Generating Samples");

    TextGen generator(model, &tokenizer);

    std::cout << "\n--- Greedy Decoding ---\n" << std::endl;
    for (const auto& prompt : prompts) {
        std::cout << "Prompt: \"" << prompt.text << "\"" << std::endl;
        std::cout << generator.generate_greedy(prompt.tokens, 150) << std::endl;
        std::cout << std::string(40, '-') << "\n" << std::endl;
    }

    std::cout << "\n--- Sampling (temp=0.8) ---\n" << std::endl;
    for (const auto& prompt : prompts) {
        std::cout << "Prompt: \"" << prompt.text << "\"" << std::endl;
        std::cout << generator.generate_sample(prompt.tokens, 0.8f, 150) << std::endl;
        std::cout << std::string(40, '-') << "\n" << std::endl;
    }
}

// One --seed drives every random stream of a run. Each stream's seed is
// its historical value XOR (seed ^ kDefaultSeed), so the default seed
// reproduces runs made before --seed existed bit for bit and any other
// seed moves all of them.
constexpr std::uint32_t kDefaultSeed = 42;

struct RunSeeds {
    std::uint32_t init;    // weight initialization (Tensor::set_init_seed)
    std::uint32_t loader;  // training-window sampling and dropout masks
};

// FNV-1a, so a path hashes the same under every standard library.
std::uint32_t fnv1a(std::string_view text) {
    std::uint32_t h = 2166136261u;
    for (const char c : text) h = (h ^ static_cast<unsigned char>(c)) * 16777619u;
    return h;
}

// The training loader samples windows with replacement, so any run that
// starts from existing weights must not repeat the seed those weights were
// trained with - it would replay the exact batch sequence the checkpoint
// already saw. Resumes perturb the seed by their step position, warm
// starts by the checkpoint path. Dropout masks follow the loader seed for
// the same reason.
RunSeeds derive_seeds(std::uint32_t seed, std::optional<int> resume_step,
                      const std::string& warm_start_path) {
    const std::uint32_t delta = seed ^ kDefaultSeed;
    std::uint32_t loader = seed;
    if (resume_step) {
        loader = seed + static_cast<std::uint32_t>(*resume_step);
    } else if (!warm_start_path.empty()) {
        loader = fnv1a(warm_start_path) ^ delta;
    }
    // Before --seed the init stream was never reseeded, so its historical
    // seed is std::mt19937's default.
    const auto init = static_cast<std::uint32_t>(std::mt19937::default_seed) ^ delta;
    return {.init = init, .loader = loader};
}

struct TrainRequest {
    const Preset& preset;
    std::string corpus_path;
    // "" trains from scratch; "resume" continues an interrupted run from
    // <prefix>_resume_model.bin / _resume_state.bin (optimizer state and
    // schedule position included); any other value is a checkpoint path to
    // warm-start from - weights only, fresh optimizer and schedule.
    std::string init;
    std::uint32_t seed;
    bool via_train_fast;  // which command the resume hint should name
};

int train(const TrainRequest& request) {
    const Preset& preset = request.preset;
    const std::string& corpus_path = request.corpus_path;
    std::cout << "\ngrad.cpp Training (" << preset.name << ")\n" << std::endl;

    const std::string prefix = checkpoint_prefix(corpus_path, preset);
    const bool resume = (request.init == "resume");
    const std::string warm_start_path = resume ? "" : request.init;

    std::optional<int> resume_next_step;
    if (resume) {
        resume_next_step = training::peek_resume_step(prefix + "_resume_state.bin");
        if (!resume_next_step) {
            throw std::runtime_error("No resume state found (" + prefix
                                     + "_resume_state.bin); start a run first");
        }
        if (*resume_next_step >= preset.num_steps) {
            std::cout << "Run already completed all " << preset.num_steps
                      << " steps; nothing to resume." << std::endl;
            return 0;
        }
    }

    BPETokenizer tokenizer(preset.vocab_size);
    const bool prepared = tokenfile::exists(token_bin_path(corpus_path, preset.vocab_size, "train"))
                       && tokenfile::exists(token_bin_path(corpus_path, preset.vocab_size, "val"));
    const TrainingData data =
        prepared ? map_prepared(corpus_path, preset.vocab_size, preset.seq_length, tokenizer)
                 : encode_in_memory(corpus_path, preset.vocab_size, preset.seq_length, tokenizer);

    utils::print_section("Initializing Model");
    const training::TrainingConfig config =
        make_config(preset, tokenizer.getCurrentVocabSize(), prefix);

    const RunSeeds seeds = derive_seeds(request.seed, resume_next_step, warm_start_path);
    Tensor::set_init_seed(seeds.init);

    const Stopwatch timer;
    GPTModel model = [&]() -> GPTModel {
        if (resume) return GPTModel::load(prefix + "_resume_model.bin");
        if (!warm_start_path.empty()) {
            std::cout << "Warm start from " << warm_start_path
                      << " (weights only, fresh optimizer)" << std::endl;
            return GPTModel::load(warm_start_path);
        }
        return GPTModel(config.vocab_size, config.d_model, config.num_layers, config.num_heads,
                        config.max_len, config.dropout, preset.arch());
    }();
    const long long init_ms = timer.ms();

    if (model.getVocabSize() != config.vocab_size || model.getDModel() != config.d_model
        || model.getNumLayers() != config.num_layers || model.getNumHeads() != config.num_heads
        || model.getArch() != preset.arch()) {
        throw std::runtime_error("Checkpoint architecture does not match preset '"
                                 + std::string(preset.name) + "'");
    }

    size_t total_params = 0;
    for (const auto& p : model.getAllParameters()) total_params += p->getData().numel();

    std::cout << "Model initialized (" << init_ms << "ms)" << std::endl;
    std::cout << "Parameters: " << (static_cast<double>(total_params) / 1e6) << "M" << std::endl;

    set_dropout_seed(seeds.loader);
    DataLoader loader(data.train, config.batch_size, true, seeds.loader);
    DataLoader val_loader(data.val, config.batch_size, false);

    std::cout << "Dataset: " << data.train->size() << " train / " << data.val->size()
              << " val sequences\n" << std::endl;

    training::Trainer trainer(config, model, loader, &val_loader);
    if (resume && !trainer.load_resume_state()) {
        throw std::runtime_error("Failed to load resume state (" + prefix + "_resume_state.bin)");
    }

    // An interrupted run (SIGINT/SIGTERM) has saved its resume pair; it
    // exits 0 like a finished one. train_supervised.sh relies on that and
    // tells the two apart by whether <prefix>_final.bin exists.
    if (!trainer.train()) {
        std::cout << "\nResume with: ./build/grad "
                  << (request.via_train_fast
                          ? "train-fast " + corpus_path
                          : "train " + corpus_path + " " + std::string(preset.name))
                  << " resume"
                  << (request.seed != kDefaultSeed ? " --seed " + std::to_string(request.seed) : "")
                  << "\n" << std::endl;
        return 0;
    }

    generate_samples(model, tokenizer, sample_prompts(corpus_path, tokenizer, *data.val));

    std::cout << "\nTraining Complete!\n" << std::endl;
    return 0;
}

// Presets `grad train` accepts: all but "fast", which is train-fast's.
std::string train_preset_names() {
    std::string names;
    for (const Preset& preset : presets()) {
        if (preset.name == "fast") continue;
        if (!names.empty()) names += ", ";
        names += preset.name;
    }
    return names;
}

constexpr const char* kSeedHelp =
    "run seed for weight init, window sampling and dropout; pass the same seed to resume";

constexpr const char* kInitHelp =
    "'resume' continues an interrupted run from its resume pair; a checkpoint path "
    "warm-starts from those weights with a fresh optimizer and schedule";

}  // namespace

int run_train(const Invocation& invocation) {
    std::string corpus = kDefaultCorpus;
    std::string preset_name = "small";
    std::string init;
    std::uint32_t seed = kDefaultSeed;

    Command cmd(invocation.usage_name(), std::string(invocation.summary));
    cmd.describe("Uses the corpus's prepared .bin token files when they exist (see prepare), "
                 "else encodes it in memory. Ctrl-C, or SIGTERM, stops at the next step and "
                 "saves <prefix>_resume_model.bin and _resume_state.bin, which are also "
                 "refreshed at every eval interval.");
    cmd.optional("corpus", corpus, "plain-text corpus");
    cmd.optional("preset", preset_name, "one of " + train_preset_names() + "; see grad presets");
    cmd.optional("init", init, kInitHelp).metavar("CKPT|resume");
    cmd.option("--seed", seed, kSeedHelp);
    if (cmd.parse(invocation.args) == ParseResult::HelpShown) return 0;

    const Preset* preset = find_preset(preset_name);
    if (!preset || preset->name == "fast") {
        throw UsageError("unknown preset '" + preset_name + "' (available: "
                         + train_preset_names() + ")");
    }
    return train({.preset = *preset,
                  .corpus_path = corpus,
                  .init = init,
                  .seed = seed,
                  .via_train_fast = false});
}

int run_train_fast(const Invocation& invocation) {
    std::string corpus = kDefaultCorpus;
    std::string init;
    std::uint32_t seed = kDefaultSeed;

    Command cmd(invocation.usage_name(), std::string(invocation.summary));
    cmd.describe("Runs train with the 'fast' preset: a 2-layer model for 50 steps, about a minute. "
                 "Checkpoints are prefixed <corpus>_fast so they never overwrite a real run.");
    cmd.optional("corpus", corpus, "plain-text corpus");
    cmd.optional("init", init, kInitHelp).metavar("CKPT|resume");
    cmd.option("--seed", seed, kSeedHelp);
    if (cmd.parse(invocation.args) == ParseResult::HelpShown) return 0;

    return train({.preset = *find_preset("fast"),
                  .corpus_path = corpus,
                  .init = init,
                  .seed = seed,
                  .via_train_fast = true});
}

}  // namespace grad::cli
