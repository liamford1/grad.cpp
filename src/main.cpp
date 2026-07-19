#include "transformer/gpt_model.h"
#include "transformer/text_gen.h"
#include "tokenizer/bpe_tokenizer.h"
#include "data/dataset.h"
#include "data/dataloader.h"
#include "data/token_file.h"
#include "training/trainer.h"
#include "utils/dashboard.h"
#include "utils/metrics.h"
#include "utils/training_utils.h"
#include <cstdlib>
#include <functional>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

std::string read_text_file(const std::string& data_path) {
    std::cout << "Reading " << data_path << "..." << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
    std::ifstream file(data_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open " + data_path);
    }
    std::string text((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
    file.close();
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << " " << (text.size() / 1024) << "KB (" << ms << "ms)" << std::endl;
    return text;
}

void load_tokenizer(const std::string& text,
                    const std::string& cache_prefix,
                    int vocab_size,
                    BPETokenizer& tokenizer) {
    std::string cache_file = cache_prefix + "_" + std::to_string(vocab_size) + ".cache";
    std::ifstream cache_check(cache_file);

    if (cache_check.good()) {
        cache_check.close();
        std::cout << "Loading tokenizer from cache..." << std::flush;
        auto start = std::chrono::high_resolution_clock::now();
        tokenizer.load(cache_file);
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << " done (" << ms << "ms)" << std::endl;
    } else {
        std::cout << "Training new tokenizer..." << std::flush;
        auto start = std::chrono::high_resolution_clock::now();
        tokenizer.train(text);
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << " done (" << ms << "ms)" << std::endl;
        tokenizer.save(cache_file);
        std::cout << "Cached to " << cache_file << std::endl;
    }

    std::cout << "Vocab size: " << tokenizer.getCurrentVocabSize() << std::endl;
}

void load_data_and_tokenizer(const std::string& data_path,
                              const std::string& cache_prefix,
                              int vocab_size,
                              std::string& text,
                              BPETokenizer& tokenizer,
                              std::vector<int>& tokens) {
    utils::print_section("Loading Data");

    text = read_text_file(data_path);
    load_tokenizer(text, cache_prefix, vocab_size, tokenizer);

    std::cout << "Encoding text..." << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
    tokens = tokenizer.encode(text);
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << " " << tokens.size() << " tokens (" << ms << "ms)" << std::endl;
}

void generate_samples(GPTModel& model, BPETokenizer& tokenizer) {
    utils::print_section("Generating Samples");

    TextGen generator(model, &tokenizer);

    std::vector<std::string> prompts = {
        "ROMEO:\n",
        "JULIET:\n",
        "First Citizen:\n"
    };

    std::cout << "\n--- Greedy Decoding ---\n" << std::endl;
    for (const auto& prompt_str : prompts) {
        std::cout << "Prompt: \"" << prompt_str << "\"" << std::endl;
        auto prompt = tokenizer.encode(prompt_str);
        std::string generated = generator.generate_greedy(prompt, 150);
        std::cout << generated << std::endl;
        std::cout << std::string(40, '-') << "\n" << std::endl;
    }

    std::cout << "\n--- Sampling (temp=0.8) ---\n" << std::endl;
    for (const auto& prompt_str : prompts) {
        std::cout << "Prompt: \"" << prompt_str << "\"" << std::endl;
        auto prompt = tokenizer.encode(prompt_str);
        std::string generated = generator.generate_sample(prompt, 0.8f, 150);
        std::cout << generated << std::endl;
        std::cout << std::string(40, '-') << "\n" << std::endl;
    }
}

// The default corpus keeps its historical cache name so existing caches
// and checkpoints stay valid; other corpora get corpus-derived names.
std::string tokenizer_cache_prefix(const std::string& corpus_path) {
    if (corpus_path == "data/shakespeare.txt") return "tokenizer";
    return corpus_path + ".tokenizer";
}

std::string token_bin_path(const std::string& corpus_path, int vocab_size,
                           const std::string& split) {
    return corpus_path + "." + std::to_string(vocab_size) + "." + split + ".bin";
}

// Inference-time tokenizer loading: the cache must exist (train/prepare
// created it), so the corpus text - possibly gigabytes - is never read.
// Falls back to training from text only for small unprepared corpora.
void load_tokenizer_for_inference(const std::string& corpus_path, int vocab_size,
                                  BPETokenizer& tokenizer) {
    std::string cache_file = tokenizer_cache_prefix(corpus_path) + "_"
                           + std::to_string(vocab_size) + ".cache";
    std::ifstream cache_check(cache_file);
    if (cache_check.good()) {
        cache_check.close();
        tokenizer.load(cache_file);
        std::cout << "Tokenizer: " << cache_file << " (vocab "
                  << tokenizer.getCurrentVocabSize() << ")" << std::endl;
        return;
    }
    std::string text = read_text_file(corpus_path);
    load_tokenizer(text, tokenizer_cache_prefix(corpus_path), vocab_size, tokenizer);
}

int run_generation(const std::string& checkpoint_path, const std::string& prompt,
                   const std::string& corpus_path, int vocab_size) {
    std::cout << "\nTransformer Generation\n" << std::endl;

    try {
        BPETokenizer tokenizer(vocab_size);
        load_tokenizer_for_inference(corpus_path, vocab_size, tokenizer);

        utils::print_section("Loading Model");
        GPTModel model = GPTModel::load(checkpoint_path);

        TextGen generator(model, &tokenizer);
        auto prompt_tokens = tokenizer.encode(prompt);

        std::cout << "\n--- Greedy Decoding ---\n" << std::endl;
        std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
        std::cout << generator.generate_greedy(prompt_tokens, 150) << std::endl;

        std::cout << "\n--- Sampling (temp=0.8) ---\n" << std::endl;
        std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
        std::cout << generator.generate_sample(prompt_tokens, 0.8f, 150) << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}

// Interactive REPL: type a prompt, watch the model continue it token by
// token. Note this is a base language model, not an instruction-tuned
// assistant: it continues text in the style of its training corpus rather
// than answering questions.
int run_chat(const std::string& checkpoint_path,
             const std::string& corpus_path, int vocab_size) {
    std::cout << "\nTransformer Chat\n" << std::endl;

    try {
        BPETokenizer tokenizer(vocab_size);
        load_tokenizer_for_inference(corpus_path, vocab_size, tokenizer);

        utils::print_section("Loading Model");
        GPTModel model = GPTModel::load(checkpoint_path);
        TextGen generator(model, &tokenizer);

        std::cout << "\nThis is a base language model: it continues text in the"
                  << " style of its training corpus.\nEmpty line or 'exit' quits.\n" << std::endl;

        std::string line;
        while (true) {
            std::cout << "> " << std::flush;
            if (!std::getline(std::cin, line)) break;
            if (line.empty() || line == "exit" || line == "quit") break;

            auto prompt = tokenizer.encode(line + "\n");
            std::cout << line << std::flush;
            generator.generate_stream(
                prompt,
                [](const std::string& piece) { std::cout << piece << std::flush; },
                0.8f, 200);
            std::cout << "\n" << std::endl;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}

// Repeatable performance benchmark: times raw training steps and token
// generation at the full model config, without writing checkpoints.
int run_benchmark(int bench_steps) {
    std::cout << "\nTransformer Benchmark\n" << std::endl;

    try {
        const int vocab_size = 5000;
        const int warmup_steps = 3;

        BPETokenizer tokenizer(vocab_size);
        std::string text = read_text_file("data/shakespeare.txt");
        load_tokenizer(text, "tokenizer", vocab_size, tokenizer);
        std::vector<int> tokens = tokenizer.encode(text);

        const int d_model = 512, num_layers = 6, num_heads = 8;
        const int max_len = 1024, seq_length = 96, batch_size = 8;

        GPTModel model(vocab_size, d_model, num_layers, num_heads, max_len, 0.1f);
        auto params = model.getAllParameters();
        AdamOptimizer optimizer(params, 3e-4f, 0.9f, 0.999f, 1e-8f, 0.0f);

        auto dataset = std::make_shared<TextDataset>(tokens, seq_length);
        DataLoader loader(dataset, batch_size, true);

        utils::print_section("Training throughput");
        std::cout << "Config: d_model=" << d_model << " layers=" << num_layers
                  << " heads=" << num_heads << " seq=" << seq_length
                  << " batch=" << batch_size << std::endl;

        auto run_step = [&]() {
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

        for (int i = 0; i < warmup_steps; i++) run_step();

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < bench_steps; i++) run_step();
        auto end = std::chrono::high_resolution_clock::now();
        double train_s = std::chrono::duration<double>(end - start).count();

        double steps_per_s = bench_steps / train_s;
        double tokens_per_s = steps_per_s * batch_size * seq_length;
        std::cout << bench_steps << " steps in " << train_s << "s" << std::endl;
        std::cout << "  " << steps_per_s << " steps/s" << std::endl;
        std::cout << "  " << tokens_per_s << " tokens/s" << std::endl;

        utils::print_section("Generation throughput");
        const int gen_tokens = 64;
        TextGen generator(model, &tokenizer);
        auto prompt = tokenizer.encode("ROMEO:\n");

        start = std::chrono::high_resolution_clock::now();
        generator.generate_sample(prompt, 0.8f, gen_tokens);
        end = std::chrono::high_resolution_clock::now();
        double gen_s = std::chrono::duration<double>(end - start).count();
        std::cout << gen_tokens << " tokens in " << gen_s << "s ("
                  << (gen_tokens / gen_s) << " tok/s)" << std::endl;

        std::cout << "\nPeak memory: " << utils::get_memory_mb() << " MB" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}

// Pre-tokenize a corpus once: train (or load) the BPE tokenizer, encode the
// whole text, and write 95/5 train/val token files. Training then memory-
// maps those files instead of re-encoding the corpus on every run.
int run_prepare(const std::string& corpus_path, int vocab_size) {
    std::cout << "\nTransformer Prepare\n" << std::endl;

    try {
        utils::print_section("Tokenizing corpus");
        std::string text = read_text_file(corpus_path);
        BPETokenizer tokenizer(vocab_size);

        // BPE merge learning scans every unique word once per merge, so its
        // cost grows with corpus size for no statistical benefit: token
        // frequencies converge long before 32MB. Train on a prefix sample
        // (cut at a word boundary), then encode the full corpus with it.
        constexpr size_t kTokenizerSampleBytes = 32ull * 1024 * 1024;
        if (text.size() > kTokenizerSampleBytes) {
            size_t cut = text.rfind(' ', kTokenizerSampleBytes);
            if (cut == std::string::npos) cut = kTokenizerSampleBytes;
            std::cout << "Corpus is " << (text.size() >> 20) << "MB; training tokenizer on a "
                      << (cut >> 20) << "MB sample" << std::endl;
            std::string sample = text.substr(0, cut);
            load_tokenizer(sample, tokenizer_cache_prefix(corpus_path), vocab_size, tokenizer);
        } else {
            load_tokenizer(text, tokenizer_cache_prefix(corpus_path), vocab_size, tokenizer);
        }

        std::cout << "Encoding text..." << std::flush;
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<int> tokens = tokenizer.encode(text);
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << " " << tokens.size() << " tokens (" << ms << "ms)" << std::endl;

        // Release the raw text before writing; the token ids are all that
        // remain live, and the split writes directly from ranges of them.
        text.clear();
        text.shrink_to_fit();

        size_t split = tokens.size() * 95 / 100;
        int vocab = tokenizer.getCurrentVocabSize();
        std::string train_bin = token_bin_path(corpus_path, vocab_size, "train");
        std::string val_bin = token_bin_path(corpus_path, vocab_size, "val");
        tokenfile::write(train_bin, tokens.data(), split, vocab);
        tokenfile::write(val_bin, tokens.data() + split, tokens.size() - split, vocab);

        std::cout << "\nWrote " << train_bin << " (" << split << " tokens)"
                  << "\nWrote " << val_bin << " (" << (tokens.size() - split) << " tokens)"
                  << "\n\nTrain with: ./build/transformer train " << corpus_path
                  << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}

// Model/training presets. "small" is the original 22M-param Shakespeare
// config; "fast" is the CI smoke test.
//
// "medium" (~70M params) is memory-bound, not compute-bound: one step at
// batch 16 / seq 256 once peaked past 10GB of phys footprint and took a
// 16GB machine down (measured 2026-07-18, BENCHMARKS.md #8). Micro-batch
// 8 with grad_accum 4 gives the optimizer an effective batch of 32 at a
// ~4GB peak (lazy grads + backward retirement, BENCHMARKS.md #9). The
// micro-batch is also measured, not assumed: 16 x accum 2 - same
// effective batch, FFN matmuls above the Metal crossover - ran 32%
// slower per token and 3GB fatter, with or without the GPU. At 40000
// steps the run consumes 40000*32*256 = 327M tokens, about one epoch of
// the TinyStories train split.
// "modern" is "medium" with the Llama-style block (RMSNorm, RoPE, SwiGLU;
// see gpt_model.h) at the same parameter count and training budget, so the
// two runs A/B the architecture and nothing else. Its checkpoints get a
// "_modern" prefix so both lineages can coexist on one corpus.
struct Preset {
    const char* name;
    int vocab_size;
    int d_model, num_layers, num_heads;
    int max_len, seq_length, batch_size, grad_accum;
    float learning_rate;
    int warmup_steps, num_steps;
    int checkpoint_interval, eval_interval;
    int max_eval_batches;  // 0 = evaluate the whole val set
    bool modern;           // GPTArch::Modern
};

// small's step count is set by measurement: on the 254K-token Shakespeare
// corpus, val perplexity bottoms around step 6000 and rises after (a 22M
// model memorizes a corpus that small). 8000 steps lets the cosine
// schedule finish near the minimum instead of training 8x past it.
const Preset kPresets[] = {
    {"fast",        500,   128, 2,  4,  1024, 64,  4, 1,  3e-4f, 10,   50,    2500, 25,  0,  false},
    {"fast-modern", 500,   128, 2,  4,  1024, 64,  4, 1,  3e-4f, 10,   50,    2500, 25,  0,  true},
    {"small",       5000,  512, 6,  8,  1024, 96,  8, 1,  3e-4f, 500,  8000,  2500, 250, 0,  false},
    {"medium",      16000, 768, 8,  12, 1024, 256, 8, 4,  3e-4f, 1000, 40000, 4000, 500, 32, false},
    {"modern",      16000, 768, 8,  12, 1024, 256, 8, 4,  3e-4f, 1000, 40000, 4000, 500, 32, true},
};

const Preset* find_preset(const std::string& name) {
    for (const auto& p : kPresets) {
        if (name == p.name) return &p;
    }
    return nullptr;
}

// Checkpoint prefix from the corpus filename: data/tinystories.txt ->
// "tinystories" (data/shakespeare.txt keeps its historical "shakespeare").
std::string checkpoint_stem(const std::string& corpus_path) {
    size_t slash = corpus_path.find_last_of('/');
    std::string base = (slash == std::string::npos) ? corpus_path
                                                    : corpus_path.substr(slash + 1);
    size_t dot = base.find_last_of('.');
    return (dot == std::string::npos) ? base : base.substr(0, dot);
}

// init_arg: "" trains from scratch; "resume" continues an interrupted run
// from <prefix>_resume_model.bin / _resume_state.bin (optimizer state and
// schedule position included); any other value is a checkpoint path to
// warm-start from - weights only, fresh optimizer and schedule.
int run_training(const Preset& preset, const std::string& corpus_path,
                 const std::string& init_arg) {
    std::cout << "\nTransformer Training (" << preset.name << ")\n" << std::endl;

    try {
        const int vocab_size = preset.vocab_size;
        const int num_steps = preset.num_steps;
        const int seq_length = preset.seq_length;
        const bool fast_mode = std::string(preset.name).rfind("fast", 0) == 0;
        const GPTArch arch = preset.modern ? GPTArch::Modern : GPTArch::GPT2;

        const std::string prefix = checkpoint_stem(corpus_path)
                                 + (preset.modern ? "_modern" : "")
                                 + (fast_mode ? "_fast" : "");
        const bool resume = (init_arg == "resume");
        const std::string warm_start_path = resume ? "" : init_arg;

        int resume_next_step = -1;
        if (resume) {
            resume_next_step = training::peek_resume_step(prefix + "_resume_state.bin");
            if (resume_next_step < 0) {
                throw std::runtime_error("No resume state found ("
                    + prefix + "_resume_state.bin); start a run first");
            }
            if (resume_next_step >= num_steps) {
                std::cout << "Run already completed all " << num_steps
                          << " steps; nothing to resume." << std::endl;
                return 0;
            }
        }

        BPETokenizer tokenizer(vocab_size);
        std::shared_ptr<Dataset> dataset;
        std::shared_ptr<Dataset> val_dataset;

        std::string train_bin = token_bin_path(corpus_path, vocab_size, "train");
        std::string val_bin = token_bin_path(corpus_path, vocab_size, "val");
        std::string cache_file = tokenizer_cache_prefix(corpus_path) + "_"
                               + std::to_string(vocab_size) + ".cache";

        if (tokenfile::exists(train_bin) && tokenfile::exists(val_bin)) {
            // Pre-tokenized path: memory-map the token files. The corpus
            // text is never loaded and nothing is re-encoded, so startup
            // cost and memory use are independent of corpus size.
            utils::print_section("Loading Data (pre-tokenized)");
            std::ifstream cache_check(cache_file);
            if (!cache_check.good()) {
                throw std::runtime_error("Found token files but no tokenizer cache ("
                                         + cache_file + "); run: ./build/transformer prepare "
                                         + corpus_path);
            }
            tokenizer.load(cache_file);

            auto mapped_train = std::make_shared<MappedTokenDataset>(train_bin, seq_length);
            auto mapped_val = std::make_shared<MappedTokenDataset>(val_bin, seq_length,
                                                                   seq_length);
            if (mapped_train->vocabSize() != tokenizer.getCurrentVocabSize()) {
                throw std::runtime_error("Token file vocab does not match tokenizer cache; re-run prepare");
            }
            std::cout << "Mapped " << mapped_train->tokenCount() << " train / "
                      << mapped_val->tokenCount() << " val tokens from "
                      << train_bin << std::endl;
            dataset = mapped_train;
            val_dataset = mapped_val;
        } else {
            // In-memory path: read and encode the corpus now. Fine for
            // small corpora; for anything large, run `prepare` first.
            std::string text;
            std::vector<int> tokens;
            load_data_and_tokenizer(corpus_path, tokenizer_cache_prefix(corpus_path),
                                    vocab_size, text, tokenizer, tokens);

            // Hold out the last 5% of the corpus for validation. The split
            // is contiguous, so no training window ever overlaps validation
            // text - val perplexity measures generalization, not
            // memorization.
            size_t split = tokens.size() * 95 / 100;
            std::vector<int> train_tokens(tokens.begin(), tokens.begin() + split);
            std::vector<int> val_tokens(tokens.begin() + split, tokens.end());

            dataset = std::make_shared<TextDataset>(train_tokens, seq_length);
            // Non-overlapping windows: evaluation covers the whole held-out
            // slice once, deterministically.
            val_dataset = std::make_shared<TextDataset>(val_tokens, seq_length,
                                                        seq_length);
        }

        utils::print_section("Initializing Model");

        training::TrainingConfig config;
        config.vocab_size = tokenizer.getCurrentVocabSize();
        config.d_model = preset.d_model;
        config.num_layers = preset.num_layers;
        config.num_heads = preset.num_heads;
        config.max_len = preset.max_len;
        config.seq_length = seq_length;
        config.batch_size = preset.batch_size;
        config.grad_accum = preset.grad_accum;
        config.learning_rate = preset.learning_rate;
        config.dropout = 0.1f;
        config.warmup_steps = preset.warmup_steps;
        config.num_steps = num_steps;
        config.checkpoint_interval = preset.checkpoint_interval;
        config.checkpoint_prefix = prefix;
        config.eval_interval = preset.eval_interval;
        config.max_eval_batches = preset.max_eval_batches;

        auto start = std::chrono::high_resolution_clock::now();
        GPTModel model = [&]() -> GPTModel {
            if (resume) return GPTModel::load(prefix + "_resume_model.bin");
            if (!warm_start_path.empty()) {
                std::cout << "Warm start from " << warm_start_path
                          << " (weights only, fresh optimizer)" << std::endl;
                return GPTModel::load(warm_start_path);
            }
            return GPTModel(config.vocab_size, config.d_model, config.num_layers,
                            config.num_heads, config.max_len, config.dropout, arch);
        }();
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (model.getVocabSize() != config.vocab_size
            || model.getDModel() != config.d_model
            || model.getNumLayers() != config.num_layers
            || model.getNumHeads() != config.num_heads
            || model.getArch() != arch) {
            throw std::runtime_error("Checkpoint architecture does not match preset '"
                                     + std::string(preset.name) + "'");
        }

        auto params = model.getAllParameters();
        int total_params = 0;
        for (const auto& p : params) total_params += p->getData().numel();

        std::cout << "Model initialized (" << ms << "ms)" << std::endl;
        std::cout << "Parameters: " << (total_params / 1e6f) << "M" << std::endl;

        // The training loader samples windows with replacement, so any run
        // that starts from existing weights must not repeat the seed those
        // weights were trained with - it would replay the exact batch
        // sequence the checkpoint already saw. Resumes perturb the seed by
        // their step position, warm starts by the checkpoint path.
        unsigned int loader_seed = 42;
        if (resume) {
            loader_seed = 42u + static_cast<unsigned int>(resume_next_step);
        } else if (!warm_start_path.empty()) {
            loader_seed = static_cast<unsigned int>(
                std::hash<std::string>{}(warm_start_path));
        }
        DataLoader loader(dataset, config.batch_size, true, loader_seed);
        DataLoader val_loader(val_dataset, config.batch_size, false);

        std::cout << "Dataset: " << dataset->size() << " train / "
                  << val_dataset->size() << " val sequences\n" << std::endl;

        training::Trainer trainer(config, model, loader, tokenizer, &val_loader);
        if (resume && !trainer.load_resume_state()) {
            throw std::runtime_error("Failed to load resume state ("
                                     + prefix + "_resume_state.bin)");
        }

        if (!trainer.train()) {
            std::cout << "\nResume with: ./build/transformer "
                      << (fast_mode ? "train-fast " + corpus_path
                                    : "train " + corpus_path + " " + preset.name)
                      << " resume\n" << std::endl;
            return 0;
        }

        generate_samples(model, tokenizer);

        std::cout << "\nTraining Complete!\n" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}

int main(int argc, char* argv[]) {
    std::string mode = (argc > 1) ? argv[1] : "";

    std::string default_corpus = "data/shakespeare.txt";

    if (mode == "train") {
        std::string corpus = (argc > 2) ? argv[2] : default_corpus;
        std::string preset_name = (argc > 3) ? argv[3] : "small";
        std::string init_arg = (argc > 4) ? argv[4] : "";
        const Preset* preset = find_preset(preset_name);
        if (!preset || preset_name == "fast") {
            std::cerr << "Unknown preset '" << preset_name << "' (available: small, medium, modern)" << std::endl;
            return 1;
        }
        return run_training(*preset, corpus, init_arg);
    }
    if (mode == "train-fast") {
        return run_training(*find_preset("fast"), (argc > 2) ? argv[2] : default_corpus,
                            (argc > 3) ? argv[3] : "");
    }
    if (mode == "prepare") {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " prepare <corpus.txt> [vocab_size]" << std::endl;
            return 1;
        }
        int vocab = (argc > 3) ? std::atoi(argv[3]) : 5000;
        return run_prepare(argv[2], vocab);
    }
    if (mode == "generate") {
        std::string checkpoint = (argc > 2) ? argv[2] : "shakespeare_final.bin";
        std::string prompt = (argc > 3) ? argv[3] : "ROMEO:\n";
        std::string corpus = (argc > 4) ? argv[4] : default_corpus;
        int vocab = (argc > 5) ? std::atoi(argv[5]) : 5000;
        return run_generation(checkpoint, prompt, corpus, vocab);
    }
    if (mode == "bench") {
        int steps = (argc > 2) ? std::atoi(argv[2]) : 20;
        return run_benchmark(steps);
    }
    if (mode == "watch") {
        // Argument is a run prefix ("tinystories_modern"), a metrics path,
        // or nothing - then the most recently modified *_metrics.csv in
        // the working directory (i.e. whatever is training right now).
        std::string target = (argc > 2) ? argv[2] : "";
        bool once = false;
        for (int i = 2; i < argc; i++) {
            if (std::string(argv[i]) == "--once") { once = true; if (target == argv[i]) target = ""; }
        }
        std::string csv;
        if (!target.empty() && target != "--once") {
            csv = (target.size() > 4 && target.substr(target.size() - 4) == ".csv")
                ? target : target + "_metrics.csv";
        } else {
            csv = utils::newest_metrics_csv(".");
            if (csv.empty()) {
                std::cerr << "No *_metrics.csv found; start a training run first "
                          << "(runs on this build log metrics automatically)." << std::endl;
                return 1;
            }
        }
        return utils::run_dashboard(csv, once);
    }
    if (mode == "chat") {
        std::string checkpoint = (argc > 2) ? argv[2] : "shakespeare_final.bin";
        std::string corpus = (argc > 3) ? argv[3] : default_corpus;
        int vocab = (argc > 4) ? std::atoi(argv[4]) : 5000;
        return run_chat(checkpoint, corpus, vocab);
    }

    std::cerr << "Usage: " << argv[0] << " <mode>\n"
              << "  prepare <corpus.txt> [vocab]         pre-tokenize a corpus to .bin token files\n"
              << "  train [corpus.txt] [small|medium|modern] [ckpt.bin|resume]\n"
              << "      full training run (uses .bin files if present); Ctrl-C saves resume\n"
              << "      state - 'resume' continues an interrupted run, a checkpoint path\n"
              << "      warm-starts from those weights with a fresh schedule\n"
              << "  train-fast [corpus.txt] [ckpt.bin|resume]   tiny config for a quick smoke test\n"
              << "  generate [ckpt] [prompt] [corpus] [vocab]   sample from a saved checkpoint\n"
              << "  chat [ckpt] [corpus] [vocab]                interactive prompt/continue REPL\n"
              << "  bench [steps]                        measure training and generation speed\n"
              << "  watch [run-prefix]                   live terminal dashboard for a training\n"
              << "      run (loss curves, val track, throughput); defaults to the most recent\n"
              << "      run in this directory - open it in a second terminal while training\n";
    return 1;
}
