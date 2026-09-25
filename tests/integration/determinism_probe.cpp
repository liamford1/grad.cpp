// Thread-count determinism probe.
//
//   determinism_probe <out-file> [reference-file]
//
// Trains a small seeded GPT of each architecture for a few AdamW steps
// (dropout on, gradient accumulation 2, clipping engaged) and writes one
// line per architecture: a 64-bit FNV-1a hash over the raw bits of every
// parameter and of every loss value seen along the way. With a reference
// file it also fails unless its output matches that file byte for byte.
//
// CMake runs it twice, at GRAD_THREADS=1 and 4. Every parallel loop
// in the engine either writes disjoint outputs or reduces in a fixed order,
// so the two runs must agree to the last bit; a reduction merged in thread
// completion order (as LayerNorm's gamma/beta sums once were) shows up here
// as a hash mismatch.

#include "grad/transformer/activations.h"
#include "grad/transformer/gpt_model.h"
#include "grad/transformer/optimizer.h"
#include "grad/transformer/parallel.h"
#include "grad/transformer/tensor.h"
#include "grad/transformer/variable.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace grad;

namespace {

constexpr int kVocab = 97;
constexpr int kDModel = 64;
constexpr int kLayers = 2;
constexpr int kHeads = 4;
constexpr size_t kSeq = 16;
constexpr size_t kBatch = 4;
constexpr float kDropout = 0.1f;
constexpr int kSteps = 5;
constexpr int kGradAccum = 2;
// Low enough that every step of this tiny model clips, so the clip path's
// global norm reduction is part of what is hashed.
constexpr float kClipNorm = 0.5f;

class Fnv1a {
public:
    void bytes(const void* p, size_t n) {
        const auto* b = static_cast<const unsigned char*>(p);
        for (size_t i = 0; i < n; i++) {
            h_ ^= b[i];
            h_ *= 0x100000001B3ull;
        }
    }
    void floats(const float* p, size_t n) { bytes(p, n * sizeof(float)); }
    [[nodiscard]] uint64_t value() const { return h_; }

private:
    uint64_t h_ = 0xCBF29CE484222325ull;
};

// One (batch, seq, 1) input/target pair of token ids, the layout the
// trainer feeds, with targets shifted by one like a language-model window.
struct TokenBatch {
    Tensor input{kBatch, kSeq, 1};
    Tensor target{kBatch, kSeq, 1};
};

TokenBatch make_batch(std::mt19937& gen) {
    std::uniform_int_distribution<int> token(0, kVocab - 1);
    TokenBatch batch;
    for (size_t b = 0; b < kBatch; b++) {
        int prev = token(gen);
        for (size_t s = 0; s < kSeq; s++) {
            const int next = token(gen);
            batch.input.setValue(b, s, 0, static_cast<float>(prev));
            batch.target.setValue(b, s, 0, static_cast<float>(next));
            prev = next;
        }
    }
    return batch;
}

float loss_of(GPTModel& model, const TokenBatch& batch, bool training) {
    auto logits = model.forward(Variable::create(batch.input, false), training);
    auto loss = logits->log_softmax()->nll_loss(Variable::create(batch.target, false));
    if (training) loss->backward();
    const float value = loss->getData().raw()[0];
    loss->release_graph();
    return value;
}

std::string probe(GPTArch arch, const char* name) {
    Tensor::set_init_seed(1234);
    set_dropout_seed(5678);
    std::mt19937 data_gen(91011);

    GPTModel model(kVocab, kDModel, kLayers, kHeads, kSeq, kDropout, arch);
    auto params = model.getAllParameters();
    AdamOptimizer optimizer(params, 3e-3f, 0.9f, 0.95f, 1e-8f, 0.1f);
    optimizer.set_schedule(/*warmup_steps=*/2, /*total_steps=*/kSteps, /*min_lr=*/3e-4f);

    Fnv1a hash;
    std::ostringstream losses;
    for (int step = 0; step < kSteps; step++) {
        optimizer.zero_grad();
        for (int micro = 0; micro < kGradAccum; micro++) {
            const float loss = loss_of(model, make_batch(data_gen), /*training=*/true);
            hash.floats(&loss, 1);
            losses << ' ' << loss;
        }
        optimizer.scale_grads(1.0f / kGradAccum);
        optimizer.clip_grad_norm(kClipNorm);
        optimizer.step();
    }

    // A forward-only pass after training covers the evaluation path too.
    const float eval_loss = loss_of(model, make_batch(data_gen), /*training=*/false);
    hash.floats(&eval_loss, 1);

    for (const auto& p : params) {
        hash.floats(p->getData().raw(), p->getData().numel());
    }

    std::cout << name << " losses:" << losses.str() << "  eval: " << eval_loss << "\n";
    char line[64];
    std::snprintf(line, sizeof(line), "%s %016llx\n", name,
                  static_cast<unsigned long long>(hash.value()));
    return line;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: " << argv[0] << " <out-file> [reference-file]\n";
        return 2;
    }

    std::string result;
    try {
        std::cout << "threads: " << ThreadPool::instance().threads() << "\n";
        result = probe(GPTArch::GPT2, "gpt2") + probe(GPTArch::Modern, "modern");
    } catch (const std::exception& e) {
        std::cerr << "determinism_probe: " << e.what() << "\n";
        return 1;
    }
    std::cout << result;

    std::ofstream out(argv[1], std::ios::binary | std::ios::trunc);
    out << result;
    out.close();
    if (!out) {
        std::cerr << "determinism_probe: cannot write " << argv[1] << "\n";
        return 1;
    }

    if (argc == 3) {
        std::ifstream ref_file(argv[2], std::ios::binary);
        if (!ref_file) {
            std::cerr << "determinism_probe: cannot read reference " << argv[2] << "\n";
            return 1;
        }
        const std::string reference((std::istreambuf_iterator<char>(ref_file)),
                                    std::istreambuf_iterator<char>());
        if (reference != result) {
            std::cerr << "FAIL: parameters differ from the reference run\n"
                      << "reference:\n" << reference << "this run:\n" << result;
            return 1;
        }
        std::cout << "matches " << argv[2] << "\n";
    }
    return 0;
}
