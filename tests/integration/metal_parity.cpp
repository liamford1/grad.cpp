// End-to-end checks of Metal-resident training (docs/design/metal-resident.md).
//
//   metal_parity parity      tiny GPT-2 and modern models trained from one seed
//                            on the CPU and on Metal: losses and parameters agree
//                            within tolerance, a second Metal run is bitwise
//                            identical, and a Metal step waits for the GPU once
//   metal_parity trainer DIR training::Trainer for a few steps in both modes,
//                            writing its files under DIR; logged losses agree
//   metal_parity gradcheck   numerical vs analytical gradients through GPU ops
//   metal_parity inference   the Metal forward (3D and 2D input) against the
//                            KV-cached CPU decoder reading the same Metal-mode
//                            weights through the fenced accessors
//
// Exits 77 (skipped) without a Metal device.
//
// Parity tolerance. Both runs see the same weights, batches and (bitwise)
// dropout masks, so they differ only by fp32 rounding in reductions that
// sum in different orders (GEMM, norms, softmax, loss, gradient norm):
// relative differences of ~1e-7 per op, which AdamW then feeds back
// through the weights. Observed: losses within 2e-6 of each other over all
// 16 micro-batches of both architectures, i.e. a few ulps at 4.8. The
// bound is 2e-4 absolute (4e-5 relative), 100x the observed drift, while a
// wrong kernel, a mask mismatch or a missed gradient term moves the loss by
// 1e-2 or more within two steps (a dangling capture in the attention
// backward did exactly that, 3e-2 at the first updated step).
#include "grad/data/dataloader.h"
#include "grad/data/dataset.h"
#include "grad/training/trainer.h"
#include "grad/transformer/activations.h"
#include "grad/transformer/device.h"
#include "grad/transformer/gpt_model.h"
#include "grad/transformer/inference.h"
#include "grad/transformer/layer_norm.h"
#include "grad/transformer/metal_backend.h"
#include "grad/transformer/optimizer.h"
#include "grad/transformer/tensor.h"
#include "grad/transformer/variable.h"
#include "../test_util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
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
constexpr int kSteps = 8;
constexpr int kGradAccum = 2;
constexpr float kClipNorm = 0.5f;  // low enough that every step clips
constexpr float kLearningRate = 3e-3f;
constexpr float kLossTolerance = 2e-4f;

struct TokenBatch {
    Tensor input;
    Tensor target;
};

TokenBatch make_batch(std::mt19937& gen) {
    std::uniform_int_distribution<int> token(0, kVocab - 1);
    TokenBatch batch{Tensor(kBatch, kSeq, 1), Tensor(kBatch, kSeq, 1)};
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

struct RunResult {
    std::vector<float> losses;      // one per micro-batch
    std::vector<float> grad_norms;  // one per step, pre-clip
    float eval_loss = 0.0f;
    std::vector<float> params;  // every parameter, concatenated
    uint64_t syncs_per_step = 0;
};

// The trainer's step, run by hand so the test sees every loss and the
// stream counters: zero, micro-batches, average, clip, AdamW, then read.
RunResult train(GPTArch arch, Device device) {
    set_device(device);
    Tensor::set_init_seed(1234);
    set_dropout_seed(5678);
    std::mt19937 data_gen(91011);

    GPTModel model(kVocab, kDModel, kLayers, kHeads, static_cast<int>(kSeq), kDropout, arch);
    auto params = model.getAllParameters();
    AdamOptimizer optimizer(params, kLearningRate, 0.9f, 0.95f, 1e-8f, 0.1f);
    optimizer.set_schedule(/*warmup_steps=*/2, /*total_steps=*/kSteps, /*min_lr=*/3e-4f);

    RunResult r;
    for (int step = 0; step < kSteps; step++) {
        const uint64_t syncs_before = metal::stream_stats().syncs;
        optimizer.zero_grad();
        std::vector<std::shared_ptr<Variable>> losses;
        for (int micro = 0; micro < kGradAccum; micro++) {
            const TokenBatch batch = make_batch(data_gen);
            auto loss = model.forward(Variable::create(batch.input), true)
                            ->cross_entropy(Variable::create(batch.target));
            loss->backward();
            loss->release_graph();
            losses.push_back(loss);
        }
        optimizer.scale_grads(1.0f / kGradAccum);
        optimizer.clip_grad_norm(kClipNorm);
        optimizer.step();
        for (const auto& loss : losses) r.losses.push_back(loss->getData().raw()[0]);
        r.grad_norms.push_back(optimizer.last_grad_norm());
        r.syncs_per_step = std::max(r.syncs_per_step, metal::stream_stats().syncs - syncs_before);
    }
    {
        NoGradGuard no_grad;
        const TokenBatch batch = make_batch(data_gen);
        r.eval_loss = model.forward(Variable::create(batch.input), false)
                          ->cross_entropy(Variable::create(batch.target))
                          ->getData()
                          .raw()[0];
    }
    for (const auto& p : params) {
        const auto v = p->getData().values();
        r.params.insert(r.params.end(), v.begin(), v.end());
    }
    set_device(Device::CPU);
    return r;
}

int run_parity() {
    for (const GPTArch arch : {GPTArch::GPT2, GPTArch::Modern}) {
        const char* name = arch == GPTArch::GPT2 ? "gpt2" : "modern";
        std::cout << "=== " << name << " ===" << std::endl;
        const RunResult cpu = train(arch, Device::CPU);
        const RunResult gpu = train(arch, Device::Metal);
        const RunResult gpu2 = train(arch, Device::Metal);

        float worst_loss = 0.0f;
        std::cout << "  micro-batch   cpu loss     metal loss   |diff|" << std::endl;
        for (size_t i = 0; i < cpu.losses.size(); i++) {
            const float diff = std::abs(cpu.losses[i] - gpu.losses[i]);
            worst_loss = std::max(worst_loss, diff);
            std::cout << "  " << std::setw(6) << i << std::fixed << std::setprecision(6)
                      << std::setw(14) << cpu.losses[i] << std::setw(14) << gpu.losses[i]
                      << std::scientific << std::setprecision(2) << std::setw(11) << diff
                      << std::defaultfloat << std::endl;
        }
        const float eval_diff = std::abs(cpu.eval_loss - gpu.eval_loss);
        float worst_norm = 0.0f;
        for (size_t i = 0; i < cpu.grad_norms.size(); i++) {
            worst_norm = std::max(
                worst_norm, std::abs(cpu.grad_norms[i] - gpu.grad_norms[i]) / cpu.grad_norms[i]);
        }
        float worst_param = 0.0f;
        size_t worst_at = 0;
        double sum_diff = 0.0;
        for (size_t i = 0; i < cpu.params.size(); i++) {
            const float diff = std::abs(cpu.params[i] - gpu.params[i]);
            sum_diff += diff;
            if (diff > worst_param) worst_param = diff, worst_at = i;
        }
        const double mean_param = sum_diff / static_cast<double>(cpu.params.size());
        std::cout << "  mean param diff " << mean_param << " (worst at index " << worst_at << ")"
                  << std::endl;
        std::cout << "  worst loss diff " << worst_loss << ", eval diff " << eval_diff
                  << ", worst grad-norm rel diff " << worst_norm << ", worst param diff "
                  << worst_param << std::endl;
        CHECK(worst_loss <= kLossTolerance);
        CHECK(eval_diff <= kLossTolerance);
        // Parameters: the mean difference is rounding level. The worst single
        // element is not bounded that tightly, because a weight whose true
        // gradient is zero or nearly so (the key biases b_k: softmax ignores a
        // constant added to a whole row of scores) receives rounding noise on
        // both sides, and Adam normalizes noise into steps of up to ~lr. One
        // lr is still ~10x below what a wrong update rule produces in 8 steps.
        CHECK(mean_param <= 1e-6);
        CHECK(worst_param <= kLearningRate);
        CHECK(worst_norm <= 1e-3f);

        const bool deterministic =
            gpu.losses == gpu2.losses
            && std::memcmp(gpu.params.data(), gpu2.params.data(), gpu.params.size() * 4) == 0;
        std::cout << "  second metal run: " << (deterministic ? "bitwise identical" : "DIFFERENT")
                  << std::endl;
        CHECK(deterministic);
        std::cout << "  metal syncs per step: " << gpu.syncs_per_step << std::endl;
        CHECK(gpu.syncs_per_step == 1);
    }
    return test_util::exit_code();
}

// The Trainer's per-step losses from its metrics CSV ("t,<step>,<loss>,...").
std::vector<float> logged_losses(const std::string& csv) {
    std::ifstream in(csv);
    std::vector<float> out;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.starts_with("t,")) continue;
        std::stringstream fields(line);
        std::string field;
        std::getline(fields, field, ',');
        std::getline(fields, field, ',');
        std::getline(fields, field, ',');
        out.push_back(std::stof(field));
    }
    return out;
}

std::vector<float> run_trainer(Device device, const std::string& prefix) {
    set_device(device);
    Tensor::set_init_seed(99);
    set_dropout_seed(7);
    std::mt19937 gen(3);
    std::uniform_int_distribution<int> token(0, kVocab - 1);
    std::vector<int> tokens(4000);
    for (int& t : tokens) t = token(gen);
    auto dataset = std::make_shared<TextDataset>(tokens, static_cast<int>(kSeq));
    DataLoader loader(dataset, static_cast<int>(kBatch), true, 11);

    training::TrainingConfig config;
    config.vocab_size = kVocab;
    config.d_model = kDModel;
    config.num_layers = kLayers;
    config.num_heads = kHeads;
    config.max_len = static_cast<int>(kSeq);
    config.seq_length = static_cast<int>(kSeq);
    config.batch_size = static_cast<int>(kBatch);
    config.learning_rate = 3e-3f;
    config.dropout = kDropout;
    config.warmup_steps = 1;
    config.num_steps = 4;
    config.checkpoint_interval = 1000;
    config.checkpoint_prefix = prefix;
    config.grad_accum = 2;
    config.eval_interval = 0;

    GPTModel model(kVocab, kDModel, kLayers, kHeads, static_cast<int>(kSeq), kDropout,
                   GPTArch::Modern);
    {
        // The metrics CSV is complete once the Trainer (its writer) is gone.
        training::Trainer trainer(config, model, loader);
        CHECK(trainer.train());
    }
    set_device(Device::CPU);
    return logged_losses(prefix + "_metrics.csv");
}

int run_trainer_parity(const std::string& dir) {
    const std::vector<float> cpu = run_trainer(Device::CPU, dir + "/metal_parity_cpu");
    const std::vector<float> gpu = run_trainer(Device::Metal, dir + "/metal_parity_metal");
    CHECK(cpu.size() == 4 && gpu.size() == 4);
    for (size_t i = 0; i < std::min(cpu.size(), gpu.size()); i++) {
        std::cout << "step " << i << std::setprecision(6) << ": cpu " << cpu[i] << ", metal "
                  << gpu[i] << std::endl;
        // The CSV rounds losses to 6 digits; the bound is the parity bound.
        CHECK_NEAR(gpu[i], cpu[i], kLossTolerance);
    }
    return test_util::exit_code();
}

// Central differences on a small graph run entirely on the GPU: matmul,
// bias add, GELU, LayerNorm, SiLU gate, dropout and the fused loss. In fp32
// a step of 1e-2 keeps truncation error (~h^2) and cancellation error
// (~eps * |L| / h, ~1e-5) both near 1e-4, so agreement is judged at 2e-2
// relative plus 2e-3 absolute: an error in a backward kernel is a
// wrong-sign or order-one discrepancy.
int run_gradcheck() {
    set_device(Device::Metal);
    std::mt19937 gen(17);
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
    const auto param = [&](size_t rows, size_t cols) {
        Tensor t = Tensor::uninitialized(rows, cols);
        for (float& v : t.values()) v = dis(gen);
        return Variable::create(std::move(t), true);
    };
    auto x = param(6, 8);
    auto w = param(8, 12);
    auto b = param(1, 12);
    auto wg = param(12, 12);
    Tensor targets = Tensor::uninitialized(6, 1);
    for (size_t i = 0; i < 6; i++) targets.raw()[i] = static_cast<float>((i * 5) % 12);
    auto tgt = Variable::create(targets);
    LayerNorm norm(12);
    set_dropout_seed(42);

    const auto loss_fn = [&] {
        // Same dropout masks on every evaluation.
        set_dropout_seed(42);
        auto h = x->matmul(w)->add(b)->gelu();
        auto n = norm.forward(h);
        auto gated = n->mul(n->matmul(wg)->silu())->dropout(0.2f, true);
        return gated->cross_entropy(tgt);
    };

    auto loss = loss_fn();
    loss->backward();
    int checked = 0;
    for (const auto& v : {x, w, b, wg, norm.getGamma(), norm.getBeta()}) {
        const std::vector<float> analytic(v->getGrad().values().begin(),
                                          v->getGrad().values().end());
        const size_t n = v->getData().numel();
        for (size_t i = 0; i < n; i += std::max<size_t>(1, n / 7)) {
            constexpr float h = 1e-2f;
            const float orig = v->getData().raw()[i];
            v->getData().raw()[i] = orig + h;
            const float plus = loss_fn()->getData().raw()[0];
            v->getData().raw()[i] = orig - h;
            const float minus = loss_fn()->getData().raw()[0];
            v->getData().raw()[i] = orig;
            const float numeric = (plus - minus) / (2.0f * h);
            const float tol = 2e-3f + 2e-2f * std::abs(numeric);
            if (!CHECK(std::abs(numeric - analytic[i]) <= tol)) {
                std::cerr << "  element " << i << ": analytic " << analytic[i] << ", numeric "
                          << numeric << std::endl;
            }
            checked++;
        }
    }
    std::cout << "checked " << checked << " gradient entries" << std::endl;
    set_device(Device::CPU);
    return test_util::exit_code();
}

// Generation stays on the CPU in Metal mode (docs/design/metal-resident.md);
// it must read the GPU-written weights correctly and agree with the GPU
// forward, as the CPU decoder agrees with the CPU forward (sanity_tests
// parity-*), within the same tolerance.
int run_inference_parity() {
    for (const GPTArch arch : {GPTArch::GPT2, GPTArch::Modern}) {
        set_device(Device::Metal);
        Tensor::set_init_seed(77);
        GPTModel model(kVocab, kDModel, kLayers, kHeads, static_cast<int>(kSeq), 0.0f, arch);
        const std::vector<int> sequence = {3, 11, 7, 0, 19, 5, 96, 42};
        const size_t S = sequence.size();
        Tensor ids3(1, S, 1);
        Tensor ids2(S, 1);
        for (size_t i = 0; i < S; i++) {
            ids3.setValue(0, i, 0, static_cast<float>(sequence[i]));
            ids2.setValue(i, 0, static_cast<float>(sequence[i]));
        }
        auto logits3 = model.forward(Variable::create(ids3), false);
        auto logits2 = model.forward(Variable::create(ids2), false);

        InferenceSession session(model);
        float worst = 0.0f;
        for (size_t i = 0; i < S; i++) {
            const float* step = session.step(sequence[i]);
            for (size_t v = 0; v < static_cast<size_t>(kVocab); v++) {
                for (const float a :
                     {logits3->getData().getValue(0, i, v), logits2->getData().getValue(i, v)}) {
                    const float tol = 1e-3f + 1e-3f * (std::abs(a) + std::abs(step[v]));
                    worst = std::max(worst, std::abs(a - step[v]) / tol);
                }
            }
        }
        std::cout << (arch == GPTArch::GPT2 ? "gpt2" : "modern") << ": worst logit disagreement "
                  << worst << " of tolerance" << std::endl;
        CHECK(worst < 1.0f);
        set_device(Device::CPU);
    }
    return test_util::exit_code();
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "usage: metal_parity parity|trainer DIR|gradcheck|inference" << std::endl;
        return 2;
    }
    if (!metal::resident_available()) {
        std::cout << "Resident Metal unavailable (" << metal::resident_status() << "), skipping."
                  << std::endl;
        return 77;
    }
    const std::string mode = argv[1];
    try {
        if (mode == "parity") return run_parity();
        if (mode == "trainer" && argc == 3) return run_trainer_parity(argv[2]);
        if (mode == "gradcheck") return run_gradcheck();
        if (mode == "inference") return run_inference_parity();
    } catch (const std::exception& e) {
        std::cerr << "metal_parity: " << e.what() << std::endl;
        return 1;
    }
    std::cerr << "usage: metal_parity parity|trainer DIR|gradcheck|inference" << std::endl;
    return 2;
}
