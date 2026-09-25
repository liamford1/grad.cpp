#include "grad/transformer/gpt_model.h"
#include "grad/transformer/inference.h"
#include "grad/transformer/variable.h"
#include "grad/transformer/optimizer.h"
#include "grad/data/dataset.h"
#include "grad/data/dataloader.h"
#include "grad/data/token_file.h"
#include "grad/tokenizer/bpe_tokenizer.h"
#include "grad/utils/training_utils.h"
#include "grad/utils/metrics.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>

using namespace grad;

void test_overfit_tiny_sequence() {
    utils::print_header("Overfitting Test: Memorize 5 Tokens");

    const int vocab_size = 20;
    const int d_model = 32;
    const int num_layers = 2;
    const int num_heads = 4;
    const int max_len = 10;
    const size_t seq_length = 5;
    const size_t vocab = vocab_size;

    GPTModel model(vocab_size, d_model, num_layers, num_heads, max_len, 0.0f);
    auto params = model.getAllParameters();
    AdamOptimizer optimizer(params, 0.001f, 0.9f, 0.999f, 1e-8f, 0.0f);

    std::vector<int> sequence = {1, 2, 3, 4, 5};
    Tensor input(seq_length, 1), target(seq_length, 1);
    for (size_t i = 0; i < seq_length; i++) {
        input.setValue(i, 0, float(sequence[i]));
        target.setValue(i, 0, float(sequence[i]));
    }

    std::cout << "Target: ";
    for (int t : sequence) std::cout << t << " ";
    std::cout << "\n\nTraining..." << std::endl;

    std::cout << std::setw(10) << "Step" << std::setw(15) << "Loss" << std::setw(20) << "Grad Norm"
              << std::endl;
    std::cout << std::string(45, '-') << std::endl;

    for (int step = 0; step < 100; step++) {
        auto in = Variable::create(input, false);
        auto tgt = Variable::create(target, false);

        auto logits = model.forward(in, true);
        auto loss = logits->log_softmax()->nll_loss(tgt);

        optimizer.zero_grad();
        loss->backward();
        float grad_norm = utils::compute_grad_norm(params);
        optimizer.clip_grad_norm(1.0f);
        optimizer.step();

        if (step % 10 == 0 || step < 5) {
            std::cout << std::setw(10) << step << std::setw(15) << std::fixed
                      << std::setprecision(6) << loss->getData().getValue(0, 0) << std::setw(20)
                      << std::fixed << std::setprecision(4) << grad_norm << std::endl;
        }

        if (step == 0 && grad_norm < 1e-6f) {
            throw std::runtime_error("Overfit test produced zero gradients");
        }
    }

    auto final_logits = model.forward(Variable::create(input, false), false);
    int correct = 0;
    for (size_t i = 0; i < seq_length; i++) {
        int pred = 0;
        float max_val = -1e9f;
        for (size_t j = 0; j < vocab; j++) {
            float val = final_logits->getData().getValue(i, j);
            if (val > max_val) {
                max_val = val;
                pred = static_cast<int>(j);
            }
        }
        if (pred == sequence[i]) correct++;
    }

    float acc = 100.0f * static_cast<float>(correct) / static_cast<float>(seq_length);
    std::cout << "\nAccuracy: " << acc << "%" << std::endl;
    if (acc < 80.0f) {
        throw std::runtime_error("Overfit test failed to reach 80% accuracy");
    }
    std::cout << "SUCCESS" << std::endl;
}

void test_dataloader() {
    utils::print_header("DataLoader Test: Multi-Batch Training");

    std::vector<int> tokens;
    for (int i = 0; i < 200; i++) tokens.push_back(i % 15);

    GPTModel model(15, 24, 2, 4, 16);
    auto dataset = std::make_shared<TextDataset>(tokens, 8);
    DataLoader loader(dataset, 2, true);

    auto params = model.getAllParameters();
    AdamOptimizer optimizer(params, 1e-5f, 0.9f, 0.999f, 1e-8f, 0.0f);

    for (int epoch = 0; epoch < 3; epoch++) {
        std::cout << "\nEpoch " << (epoch + 1) << std::endl;
        loader.reset();
        float total_loss = 0.0f;
        int batch_count = 0;

        while (loader.has_next()) {
            auto batch = loader.next_batch();

            const size_t batch_size = batch.input.getBatchSize();
            const size_t seq_len = batch.input.getRows();

            Tensor input_2d(batch_size * seq_len, 1);
            Tensor target_2d(batch_size * seq_len, 1);
            utils::reshape_batch_to_2d(batch.input, batch.target, input_2d, target_2d);

            auto in = Variable::create(input_2d, false);
            auto tgt = Variable::create(target_2d, false);

            auto loss = model.forward(in, true)->log_softmax()->nll_loss(tgt);

            optimizer.zero_grad();
            loss->backward();
            loss->release_graph();
            optimizer.clip_grad_norm(1.0f);
            optimizer.step();

            total_loss += loss->getData().getValue(0, 0);
            batch_count++;

            if (batch_count % 10 == 0) {
                std::cout << "  Batch " << batch_count << " - Loss: " << std::fixed
                          << std::setprecision(4) << loss->getData().getValue(0, 0) << std::endl;
            }
        }

        const float avg_loss = total_loss / static_cast<float>(batch_count);
        std::cout << "Avg Loss: " << avg_loss << std::endl;
        if (batch_count == 0 || !std::isfinite(avg_loss)) {
            throw std::runtime_error("DataLoader test produced a non-finite loss");
        }
    }

    std::cout << "DataLoader test complete" << std::endl;
}

// Capped validation reads SpreadSubset: `count` windows at
// floor(i * size / count), first window included, the same on every read.
void test_spread_subset() {
    utils::print_header("SpreadSubset: evenly spread, deterministic windows");
    std::vector<int> tokens(1001);
    for (size_t i = 0; i < tokens.size(); i++) tokens[i] = static_cast<int>(i);
    auto source = std::make_shared<TextDataset>(tokens, 10, 10);  // 99 windows
    SpreadSubset subset(source, 7);
    if (subset.size() != 7)
        throw std::runtime_error("SpreadSubset size is not the requested count");
    for (size_t i = 0; i < subset.size(); i++) {
        const size_t expected_window = i * source->size() / subset.size();
        const int first_token = subset.get_item(i).first.front();
        if (first_token != static_cast<int>(expected_window * 10)) {
            throw std::runtime_error("SpreadSubset window " + std::to_string(i)
                                     + " starts at token " + std::to_string(first_token));
        }
    }
    if (subset.get_item(6).first.front() < 800) {
        throw std::runtime_error("SpreadSubset does not reach the end of the source");
    }
    if (SpreadSubset(source, 1000).size() != source->size()) {
        throw std::runtime_error("SpreadSubset count is not capped at the source size");
    }
    std::cout << "SUCCESS" << std::endl;
}

void benchmark_training_speed() {
    utils::print_header("Performance Benchmark: 100 Steps");

    std::ifstream file("data/shakespeare.txt");
    if (!file.is_open()) {
        std::cerr << "Can't open data/shakespeare.txt" << std::endl;
        return;
    }

    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    std::istringstream sample_stream(text);
    std::string sample_text;
    std::string word;
    int word_count = 0;
    int max_training_words = 20000;

    while (sample_stream >> word && word_count < max_training_words) {
        sample_text += word + " ";
        word_count++;
    }

    std::cout << "Training BPE tokenizer on " << word_count << " words (sampled)..." << std::endl;
    BPETokenizer tokenizer(3000);
    tokenizer.train(sample_text);
    std::cout << "Tokenizer trained! Encoding full text..." << std::endl;
    std::vector<int> tokens = tokenizer.encode(text);
    std::cout << "Encoded " << tokens.size() << " tokens." << std::endl;

    const int vocab_size = tokenizer.getCurrentVocabSize();
    const int d_model = 512;
    const int num_layers = 6;
    const int num_heads = 8;
    const int max_len = 1024;
    const int seq_length = 128;
    const int batch_size = 8;

    GPTModel model(vocab_size, d_model, num_layers, num_heads, max_len, 0.1f);
    auto params = model.getAllParameters();
    auto dataset = std::make_shared<TextDataset>(tokens, seq_length);
    DataLoader loader(dataset, batch_size, true);
    AdamOptimizer optimizer(params, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f);

    std::cout << "Config: vocab=" << vocab_size << " d_model=" << d_model
              << " layers=" << num_layers << " heads=" << num_heads << std::endl;
    std::cout << "Running 100 steps...\n" << std::endl;

    for (int step = 0; step < 5; step++) {
        if (!loader.has_next()) loader.reset();
        auto batch = loader.next_batch();

        const size_t bs = batch.input.getBatchSize();
        const size_t sl = batch.input.getRows();
        Tensor input_2d(bs * sl, 1), target_2d(bs * sl, 1);
        utils::reshape_batch_to_2d(batch.input, batch.target, input_2d, target_2d);

        auto in = Variable::create(input_2d, false);
        auto tgt = Variable::create(target_2d, false);
        auto loss = model.forward(in, true)->log_softmax()->nll_loss(tgt);
        optimizer.zero_grad();
        loss->backward();
        loss->release_graph();
        optimizer.clip_grad_norm(5.0f);
        optimizer.step();
    }

    std::cout << "Warmup complete. Starting benchmark..." << std::endl;

    auto start = std::chrono::steady_clock::now();

    for (int step = 0; step < 100; step++) {
        if (!loader.has_next()) loader.reset();
        auto batch = loader.next_batch();

        const size_t bs = batch.input.getBatchSize();
        const size_t sl = batch.input.getRows();
        Tensor input_2d(bs * sl, 1), target_2d(bs * sl, 1);
        utils::reshape_batch_to_2d(batch.input, batch.target, input_2d, target_2d);

        auto in = Variable::create(input_2d, false);
        auto tgt = Variable::create(target_2d, false);
        auto loss = model.forward(in, true)->log_softmax()->nll_loss(tgt);
        optimizer.zero_grad();
        loss->backward();
        loss->release_graph();
        optimizer.clip_grad_norm(5.0f);
        optimizer.step();
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "\n=== BASELINE RESULTS ===" << std::endl;
    std::cout << "100 steps took: " << duration.count() << "ms" << std::endl;
    const double ms = static_cast<double>(duration.count());
    std::cout << "Average per step: " << (ms / 100.0) << "ms" << std::endl;
    std::cout << "Speed: " << (100000.0 / ms) << " steps/sec" << std::endl;
}

// The batched training forward and the KV-cache InferenceSession are two
// independent implementations of the same model; this feeds one token
// sequence through both and requires per-position logits to agree. Guards
// the class of bug where the two paths drift (e.g. a RoPE pairing or norm
// epsilon mismatch), which gradient checks cannot catch.
void test_inference_parity(GPTArch arch) {
    utils::print_header(std::string("Inference Parity: ")
                        + (arch == GPTArch::Modern ? "Modern arch" : "GPT-2 arch"));

    const int vocab_size = 23;
    const int d_model = 32;
    const int num_layers = 2;
    const int num_heads = 4;  // head_size 8, even as RoPE requires
    const int max_len = 16;
    const std::vector<int> sequence = {3, 11, 7, 0, 19, 5};
    const size_t S = sequence.size();
    const size_t vocab = vocab_size;

    GPTModel model(vocab_size, d_model, num_layers, num_heads, max_len,
                   /*dropout=*/0.0f, arch);

    // Batched training-path forward over the whole sequence.
    Tensor ids(1, S, 1);
    for (size_t i = 0; i < S; i++) ids.setValue(0, i, 0, static_cast<float>(sequence[i]));
    auto logits = model.forward(Variable::create(ids, false), false);

    // Incremental decoding over the same tokens.
    InferenceSession session(model);
    float worst = 0.0f;
    for (size_t i = 0; i < S; i++) {
        const float* step_logits = session.step(sequence[i]);
        for (size_t v = 0; v < vocab; v++) {
            float a = logits->getData().getValue(0, i, v);
            float b = step_logits[v];
            float tol = 1e-3f + 1e-3f * (std::abs(a) + std::abs(b));
            worst = std::max(worst, std::abs(a - b) / tol);
        }
    }
    logits->release_graph();

    std::cout << "Worst logit disagreement: " << worst << " of tolerance" << std::endl;
    if (worst >= 1.0f) {
        throw std::runtime_error("Inference parity FAILED");
    }
    std::cout << "SUCCESS" << std::endl;
}

// File-format robustness: corrupt token files and tokenizer caches must be
// rejected with an exception (not a crash, a leak, or a huge allocation),
// and a failed tokenizer load must leave the tokenizer as it was.
void expect_throw(const std::string& what, const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::exception& e) {
        std::cout << "  rejected " << what << ": " << e.what() << std::endl;
        return;
    }
    throw std::runtime_error("expected an exception for " + what);
}

void write_bytes(const std::string& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("could not write " + path);
}

std::string token_file_bytes(uint32_t vocab, uint64_t count, const std::vector<uint16_t>& ids,
                             const char* magic = "TOK1") {
    std::string bytes(magic, 4);
    bytes.append(reinterpret_cast<const char*>(&vocab), sizeof(vocab));
    bytes.append(reinterpret_cast<const char*>(&count), sizeof(count));
    bytes.append(reinterpret_cast<const char*>(ids.data()), ids.size() * sizeof(uint16_t));
    return bytes;
}

void test_file_formats() {
    utils::print_header("File Formats: corrupt inputs are rejected");
    const std::string tok = "sanity_io_test.bin";
    const std::vector<uint16_t> ids = {1, 2, 3, 4, 5, 6, 7, 8};

    tokenfile::write(tok, std::vector<int>(ids.begin(), ids.end()), 10);
    {
        MappedTokenDataset ok(tok, 4);
        if (ok.tokenCount() != ids.size() || ok.vocabSize() != 10 || ok.size() != 4) {
            throw std::runtime_error("valid token file read back wrong");
        }
        if (ok.get_item(3).second.back() != 8) {
            throw std::runtime_error("valid token file window read back wrong");
        }
    }
    write_bytes(tok, token_file_bytes(10, ids.size(), ids, "NOPE"));
    expect_throw("bad magic", [&] { MappedTokenDataset d(tok, 4); });
    write_bytes(tok, token_file_bytes(0, ids.size(), ids));
    expect_throw("zero vocab", [&] { MappedTokenDataset d(tok, 4); });
    write_bytes(tok, token_file_bytes(10, ids.size() + 1, ids));
    expect_throw("count past end of file", [&] { MappedTokenDataset d(tok, 4); });
    // count * sizeof(uint16_t) wraps to a small number in 64 bits.
    write_bytes(tok, token_file_bytes(10, (uint64_t{1} << 63) + 2, ids));
    expect_throw("count that overflows a byte size", [&] { MappedTokenDataset d(tok, 4); });
    write_bytes(tok, token_file_bytes(10, ids.size(), ids));
    expect_throw("window longer than the file", [&] { MappedTokenDataset d(tok, 8); });
    std::vector<uint16_t> bad_ids = ids;
    bad_ids[5] = 10;
    write_bytes(tok, token_file_bytes(10, bad_ids.size(), bad_ids));
    {
        MappedTokenDataset d(tok, 4);
        d.get_item(0);  // tokens 0..4 are in range
        expect_throw("token id >= vocab", [&] { d.get_item(1); });
    }
    std::remove(tok.c_str());

    const std::string cache = "sanity_io_test.cache";
    BPETokenizer trained(40);
    trained.train("the cat sat on the mat and the cat ran");
    trained.save(cache);
    const std::string text = "the cat sat on the mat";
    BPETokenizer loaded(40);
    loaded.load(cache);
    if (loaded.encode(text) != trained.encode(text)
        || loaded.getCurrentVocabSize() != trained.getCurrentVocabSize()) {
        throw std::runtime_error("tokenizer cache round trip changed the tokenizer");
    }

    std::ifstream in(cache, std::ios::binary);
    const std::string good((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    write_bytes(cache, good.substr(0, good.size() - 3));
    expect_throw("truncated cache", [&] { loaded.load(cache); });
    if (loaded.encode(text) != trained.encode(text)) {
        throw std::runtime_error("failed tokenizer load modified the tokenizer");
    }
    std::string huge = good;
    const uint64_t huge_len = uint64_t{1} << 40;
    huge.replace(sizeof(size_t), sizeof(huge_len), reinterpret_cast<const char*>(&huge_len),
                 sizeof(huge_len));
    write_bytes(cache, huge);
    expect_throw("1TB token length", [&] { loaded.load(cache); });
    std::remove(cache.c_str());

    std::cout << "SUCCESS" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "Running Sanity Tests\n" << std::endl;

    try {
        if (argc > 1) {
            std::string test_name(argv[1]);
            if (test_name == "overfit") {
                test_overfit_tiny_sequence();
            } else if (test_name == "dataloader") {
                test_dataloader();
                test_spread_subset();
            } else if (test_name == "benchmark") {
                benchmark_training_speed();
            } else if (test_name == "parity-gpt2") {
                test_inference_parity(GPTArch::GPT2);
            } else if (test_name == "parity-modern") {
                test_inference_parity(GPTArch::Modern);
            } else if (test_name == "file-formats") {
                test_file_formats();
            } else {
                std::cerr << "Unknown test: " << test_name << std::endl;
                std::cerr << "Available tests: overfit, dataloader, parity-gpt2, "
                             "parity-modern, file-formats, benchmark"
                          << std::endl;
                return 1;
            }
        } else {
            test_overfit_tiny_sequence();
            test_dataloader();
            test_spread_subset();
            test_inference_parity(GPTArch::GPT2);
            test_inference_parity(GPTArch::Modern);
            test_file_formats();
            benchmark_training_speed();
        }

        std::cout << "\nALL TESTS COMPLETED!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}
