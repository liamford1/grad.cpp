// Grad mode: NoGradGuard records no graph and leaves training gradients
// untouched.
//
// Uses its own CHECK rather than assert so it stays meaningful in Release
// (NDEBUG) builds.

#include "transformer/gpt_model.h"
#include "transformer/tensor.h"
#include "transformer/variable.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,    \
                         #cond);                                            \
            g_failures++;                                                   \
        }                                                                   \
    } while (0)

bool throws_logic_error(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::logic_error&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

bool same_bits(const Tensor& a, const Tensor& b) {
    if (a.numel() != b.numel()) return false;
    return a.numel() == 0 || std::memcmp(a.raw(), b.raw(), a.numel() * sizeof(float)) == 0;
}

Tensor filled(Tensor t, float phase) {
    for (size_t i = 0; i < t.numel(); i++) {
        t.raw()[i] = std::sin(0.37f * static_cast<float>(i) + phase);
    }
    return t;
}

void test_guard_scope() {
    CHECK(GradMode::is_enabled());
    {
        NoGradGuard outer;
        CHECK(!GradMode::is_enabled());
        {
            NoGradGuard inner;
            CHECK(!GradMode::is_enabled());
        }
        CHECK(!GradMode::is_enabled());

        // Thread-local: another thread still records graphs.
        bool other_enabled = false;
        std::thread([&] { other_enabled = GradMode::is_enabled(); }).join();
        CHECK(other_enabled);
    }
    CHECK(GradMode::is_enabled());
}

void test_ops_under_guard() {
    auto x = Variable::create(filled(Tensor(3, 4), 0.1f), true);
    auto y = Variable::create(filled(Tensor(3, 4), 0.2f), true);
    auto w = Variable::create(filled(Tensor(4, 2), 0.4f), true);
    auto targets = Variable::create(Tensor(3, 1), false);

    NoGradGuard no_grad;
    const std::vector<std::shared_ptr<Variable>> outputs = {
        x->matmul(w), x->add(y), x->scale(2.0f), x->softmax(), x->gelu(),
        x->silu(), x->mul(y), x->dropout(0.5f, /*training=*/true),
        x->log_softmax(), x->log_softmax()->nll_loss(targets),
    };
    for (const auto& out : outputs) CHECK(!out->requiresGrad());

    // Leaves keep their flag, and a root made under the guard cannot seed
    // a backward pass.
    CHECK(x->requiresGrad());
    CHECK(throws_logic_error([&] { outputs.back()->backward(); }));
    CHECK(!x->hasGrad() && !y->hasGrad() && !w->hasGrad());
}

float loss_value(const std::shared_ptr<Variable>& loss) { return loss->getData().raw()[0]; }

void test_model_under_guard(GPTArch arch) {
    Tensor::set_init_seed(21);
    GPTModel model(13, 16, 2, 2, 8, 0.1f, arch);
    Tensor ids(2, 8, 1), next(2, 8, 1);
    for (size_t i = 0; i < ids.numel(); i++) {
        ids.raw()[i] = static_cast<float>((5 * i + 1) % 13);
        next.raw()[i] = static_cast<float>((5 * i + 6) % 13);
    }
    auto params = model.getAllParameters();
    auto loss_of = [&](bool training) {
        return model.forward(Variable::create(ids, false), training)
            ->log_softmax()->nll_loss(Variable::create(next, false));
    };
    auto grads_of_one_step = [&] {
        for (auto& p : params) p->zeroGrad();
        auto loss = loss_of(/*training=*/false);
        loss->backward();
        std::vector<Tensor> grads;
        for (auto& p : params) grads.push_back(p->getGrad());
        return grads;
    };

    const std::vector<Tensor> before = grads_of_one_step();
    const float recorded = loss_value(loss_of(false));
    const long refs_idle = params[0].use_count();

    {
        NoGradGuard no_grad;
        for (bool training : {false, true}) {
            auto loss = loss_of(training);
            CHECK(!loss->requiresGrad());
            CHECK(throws_logic_error([&] { loss->backward(); }));
        }
        // No graph node holds a parameter while the loss is alive, and
        // the forward value is exactly the recorded one.
        auto loss = loss_of(false);
        CHECK(params[0].use_count() == refs_idle);
        CHECK(loss_value(loss) == recorded);
    }

    // With the graph recorded, the live loss does hold the parameters,
    // which is what makes the use_count probe above meaningful.
    {
        auto loss = loss_of(false);
        CHECK(params[0].use_count() > refs_idle);
    }

    const std::vector<Tensor> after = grads_of_one_step();
    bool unchanged = before.size() == after.size();
    for (size_t i = 0; unchanged && i < before.size(); i++) {
        unchanged = same_bits(before[i], after[i]);
    }
    CHECK(unchanged);
}

}  // namespace

int main() {
    test_guard_scope();
    test_ops_under_guard();
    test_model_under_guard(GPTArch::GPT2);
    test_model_under_guard(GPTArch::Modern);

    if (g_failures) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("All grad-mode checks passed\n");
    return 0;
}
