// Full-model gradient check: runs a tiny GPTModel end to end (3D batched
// input -> log_softmax -> nll_loss) and compares analytical gradients from
// backward() against central-difference numerical gradients for parameters
// in every component: embeddings, attention, FFN, and layer norms.
//
// This is the only test that exercises the weight-tying backward and the
// batched (3D) training path through the whole network.
#include "grad/transformer/gpt_model.h"
#include "grad/transformer/variable.h"
#include "grad/transformer/tensor.h"
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace grad;

namespace {

constexpr int kVocab = 11;
constexpr int kDModel = 16;
constexpr int kLayers = 2;
constexpr int kHeads = 2;
constexpr int kMaxLen = 32;
constexpr size_t kBatch = 2;
constexpr size_t kSeq = 5;

float model_loss(const GPTModel& model,
                 const std::shared_ptr<Variable>& input,
                 const std::shared_ptr<Variable>& targets) {
    auto logits = model.forward(input, /*training=*/true);  // dropout is 0
    auto loss = logits->log_softmax()->nll_loss(targets);
    float value = loss->getData().getValue(0, 0);
    loss->release_graph();
    return value;
}

float numerical_gradient(const GPTModel& model,
                         const std::shared_ptr<Variable>& input,
                         const std::shared_ptr<Variable>& targets,
                         float* param, float epsilon = 1e-2f) {
    float original = *param;
    *param = original + epsilon;
    double loss_plus = model_loss(model, input, targets);
    *param = original - epsilon;
    double loss_minus = model_loss(model, input, targets);
    *param = original;
    return static_cast<float>((loss_plus - loss_minus) / (2.0 * epsilon));
}

bool check_point(const std::string& name, float analytical, float numerical,
                 int& passed, int& total) {
    float abs_err = std::abs(analytical - numerical);
    float rel_err = abs_err /
        (std::abs(analytical) + std::abs(numerical) + 1e-8f);

    // Small gradients drown in float forward-pass noise, so accept either
    // a tight relative error or a small absolute error.
    bool ok = rel_err < 3e-2f || abs_err < 1e-4f;

    std::cout << (ok ? "  PASS " : "  FAIL ") << name
              << ": analytical=" << analytical
              << " numerical=" << numerical
              << " rel_err=" << rel_err << std::endl;
    total++;
    if (ok) passed++;
    return ok;
}

struct Probe { const char* name; int param_idx; int flat_idx; };

// Runs the end-to-end gradient check for one architecture, probing the
// given parameter entries. Probes index into getAllParameters(), whose
// layout differs by arch (see the tables in main).
void run_arch(GPTArch arch, const std::vector<Probe>& probes,
              int& passed, int& total) {
    GPTModel model(kVocab, kDModel, kLayers, kHeads, kMaxLen, /*dropout=*/0.0f, arch);

    Tensor ids(kBatch, kSeq, 1);
    Tensor tgt(kBatch, kSeq, 1);
    for (size_t b = 0; b < kBatch; b++) {
        for (size_t s = 0; s < kSeq; s++) {
            ids.setValue(b, s, 0, static_cast<float>((b * 3 + s * 2 + 1) % kVocab));
            tgt.setValue(b, s, 0, static_cast<float>((b * 5 + s * 3 + 2) % kVocab));
        }
    }
    auto input = Variable::create(ids, false);
    auto targets = Variable::create(tgt, false);

    auto params = model.getAllParameters();
    for (auto& p : params) p->zeroGrad();

    auto logits = model.forward(input, true);
    auto loss = logits->log_softmax()->nll_loss(targets);
    loss->backward();

    for (const auto& probe : probes) {
        int idx = probe.param_idx >= 0
            ? probe.param_idx
            : static_cast<int>(params.size()) + probe.param_idx;
        auto& p = params[static_cast<size_t>(idx)];
        float analytical = p->hasGrad() ? p->getGrad().raw()[probe.flat_idx] : 0.0f;
        float numerical = numerical_gradient(model, input, targets,
                                             &p->getData().raw()[probe.flat_idx]);
        check_point(probe.name, analytical, numerical, passed, total);
    }
}

}  // namespace

int main() {
    int passed = 0, total = 0;

    std::cout << "=== FULL-MODEL GRADIENT CHECK (GPT-2 arch, 3D batched path) ===" << std::endl;
    // params[0] = token embedding table (also the tied output projection),
    // params[1] = positional embeddings, then 16 per layer (8 attention,
    // W1/b1/W2/b2, two norms' gamma/beta), ending with the final norm.
    run_arch(GPTArch::GPT2, {
        {"embedding[0]",        0, 0},
        {"embedding[mid]",      0, (kVocab / 2) * kDModel + 3},
        {"pos_embedding[7]",    1, 7},
        {"layer0.W_q[5]",       2, 5},
        {"layer0.W_o[9]",       5, 9},
        {"layer0.b_q[1]",       6, 1},
        {"layer0.ff_w1[11]",   10, 11},
        {"layer0.ff_w2[4]",    12, 4},
        {"layer0.norm1.g[2]",  14, 2},
        {"layer1.W_v[8]",      2 + 16 + 2, 8},
        {"final_norm.g[3]",    -2, 3},
        {"final_norm.b[3]",    -1, 3},
    }, passed, total);

    std::cout << "\n=== FULL-MODEL GRADIENT CHECK (Modern arch: RMSNorm+RoPE+SwiGLU) ===" << std::endl;
    // No positional embeddings; 15 per layer (8 attention, gate/up/down,
    // two norms' gamma/beta - the betas exist but are inert in RMS mode,
    // so they are not probed). Covers the RoPE rotation backward, the
    // silu/mul ops, and the RMSNorm backward end to end.
    run_arch(GPTArch::Modern, {
        {"embedding[0]",        0, 0},
        {"embedding[mid]",      0, (kVocab / 2) * kDModel + 3},
        {"layer0.W_q[5]",       1, 5},
        {"layer0.W_k[7]",       2, 7},
        {"layer0.W_o[9]",       4, 9},
        {"layer0.b_q[1]",       5, 1},
        {"layer0.ff_gate[3]",   9, 3},
        {"layer0.ff_up[11]",   10, 11},
        {"layer0.ff_down[4]",  11, 4},
        {"layer0.norm1.g[2]",  12, 2},
        {"layer1.W_v[8]",      1 + 15 + 2, 8},
        {"final_norm.g[3]",    -2, 3},
    }, passed, total);

    std::cout << "\nPassed " << passed << "/" << total << std::endl;
    return (passed == total) ? 0 : 1;
}
