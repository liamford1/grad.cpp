// Regression tests for core-engine invariants: dropout RNG independence
// and determinism, nested parallel_for, tensor shape validation, loss
// gradients, checkpoint validation, and the 2D attention path.
//
// Uses its own CHECK rather than assert so it stays meaningful in Release
// (NDEBUG) builds.

#include "transformer/activations.h"
#include "transformer/gpt_model.h"
#include "transformer/multihead_attention.h"
#include "transformer/parallel.h"
#include "transformer/tensor.h"
#include "transformer/variable.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
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

template <typename E>
bool throws(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const E&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

double drop_fraction(const std::vector<float>& m) {
    size_t zeros = 0;
    for (float v : m) zeros += (v == 0.0f);
    return static_cast<double>(zeros) / static_cast<double>(m.size());
}

void test_dropout_masks() {
    const float rate = 0.1f;
    const float scale = 1.0f / (1.0f - rate);

    // Multi-block (and therefore multi-chunk, parallel) masks.
    const size_t n = (size_t{1} << 18) + 100;
    std::vector<float> a(n), b(n);
    fill_dropout_mask(a.data(), n, rate, scale);
    fill_dropout_mask(b.data(), n, rate, scale);
    CHECK(a != b);
    // sigma of the drop fraction at n = 262k is ~0.0006; 0.005 is ~8 sigma.
    CHECK(std::fabs(drop_fraction(a) - rate) < 0.005);
    CHECK(std::fabs(drop_fraction(b) - rate) < 0.005);
    for (float v : a) CHECK(v == 0.0f || v == scale);

    // Two threads filling at once must not produce the same mask. Each
    // call is below the parallel grain, so it runs on its own thread.
    const size_t small = 4096;
    std::vector<float> t1(small), t2(small);
    std::thread th1([&] { fill_dropout_mask(t1.data(), small, 0.5f, 2.0f); });
    std::thread th2([&] { fill_dropout_mask(t2.data(), small, 0.5f, 2.0f); });
    th1.join();
    th2.join();
    CHECK(t1 != t2);
    CHECK(std::fabs(drop_fraction(t1) - 0.5) < 0.05);
    CHECK(std::fabs(drop_fraction(t2) - 0.5) < 0.05);

    // Reseeding replays the sequence; a different seed changes it.
    set_dropout_seed(42);
    fill_dropout_mask(a.data(), n, rate, scale);
    set_dropout_seed(42);
    fill_dropout_mask(b.data(), n, rate, scale);
    CHECK(a == b);
    set_dropout_seed(43);
    fill_dropout_mask(b.data(), n, rate, scale);
    CHECK(a != b);

    // A mask depends on (seed, stream, position) only, not on whether the
    // fill ran across the pool or inline inside another parallel body.
    const uint64_t stream = 12345;
    fill_dropout_mask(a.data(), n, rate, scale, stream);
    std::fill(b.begin(), b.end(), -1.0f);
    parallel_for(2, 1, [&](size_t begin, size_t) {
        if (begin == 0) fill_dropout_mask(b.data(), n, rate, scale, stream);
    });
    CHECK(a == b);

    // Distinct reserved streams give distinct masks.
    const uint64_t base = reserve_dropout_streams(2);
    fill_dropout_mask(a.data(), small, rate, scale, base);
    fill_dropout_mask(b.data(), small, rate, scale, base + 1);
    CHECK(!std::equal(a.begin(), a.begin() + small, b.begin()));
}

void test_init_seed() {
    Tensor a(64, 32), b(64, 32);
    Tensor::set_init_seed(7);
    a.xavier(64, 32);
    Tensor::set_init_seed(7);
    b.xavier(64, 32);
    CHECK(std::memcmp(a.raw(), b.raw(), a.numel() * sizeof(float)) == 0);

    const float limit = std::sqrt(6.0f / (64 + 32));
    bool in_range = true;
    for (size_t i = 0; i < a.numel(); i++) {
        in_range = in_range && std::fabs(a.raw()[i]) <= limit;
    }
    CHECK(in_range);

    Tensor::set_init_seed(8);
    b.xavier(64, 32);
    CHECK(std::memcmp(a.raw(), b.raw(), a.numel() * sizeof(float)) != 0);
}

void test_parallel_for() {
    // Nested calls, including from the calling thread's own chunks, run
    // inline and still cover every index exactly once.
    std::atomic<int> count{0};
    parallel_for(64, 1, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; i++) {
            parallel_for(100, 1, [&](size_t b, size_t e) {
                count.fetch_add(static_cast<int>(e - b), std::memory_order_relaxed);
            });
        }
    });
    CHECK(count.load() == 64 * 100);

    // grain 0 behaves as grain 1.
    std::vector<std::atomic<int>> hits(1000);
    parallel_for(hits.size(), 0, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; i++) hits[i].fetch_add(1);
    });
    bool once = true;
    for (auto& h : hits) once = once && h.load() == 1;
    CHECK(once);
}

// Scalar loss sum(out * R) whose backward seeds out's grad with R.
std::shared_ptr<Variable> weighted_sum(const std::shared_ptr<Variable>& out,
                                       const std::vector<float>& R) {
    const Tensor& o = out->getData();
    double sum = 0.0;
    for (size_t i = 0; i < o.numel(); i++) sum += static_cast<double>(o.raw()[i]) * R[i];
    Tensor t(1, 1);
    t.raw()[0] = static_cast<float>(sum);
    auto loss = Variable::create(std::move(t), true);
    loss->addChild(out);
    loss->setBackwardFn([out, R]() {
        out->ensureGrad();
        float* g = out->getGrad().raw();
        for (size_t i = 0; i < R.size(); i++) g[i] += R[i];
    });
    return loss;
}

// Finite-difference check of 2D attention in training mode with dropout
// active. Reseeding before every forward pins the masks, so the function
// being differentiated is fixed.
void test_attention_2d_dropout_gradients(bool rope) {
    const int S = 5, d = 8, H = 2;
    const float rate = 0.3f;
    const uint64_t seed = 99;

    Tensor::set_init_seed(3);
    MultiHeadAttention attn(d, H, rate, rope);
    Tensor x(S, d);
    x.xavier(S, d);
    std::vector<float> R(static_cast<size_t>(S) * d);
    for (size_t i = 0; i < R.size(); i++) R[i] = std::sin(0.7f * static_cast<float>(i) + 0.3f);

    auto loss_at = [&](const Tensor& xin) {
        set_dropout_seed(seed);
        auto in = Variable::create(xin, false);
        auto out = attn.forward(in, /*training=*/true);
        double sum = 0.0;
        for (size_t i = 0; i < R.size(); i++) sum += static_cast<double>(out->getData().raw()[i]) * R[i];
        out->release_graph();
        return sum;
    };

    set_dropout_seed(seed);
    auto input = Variable::create(x, true);
    auto out = attn.forward(input, /*training=*/true);
    CHECK(!out->getData().getIs3D());
    CHECK(out->getData().getRows() == static_cast<size_t>(S));

    // The 2D result equals the batch-1 3D result under the same masks.
    {
        set_dropout_seed(seed);
        Tensor x3(1, S, d);
        std::memcpy(x3.raw(), x.raw(), x.numel() * sizeof(float));
        auto out3 = attn.forward(Variable::create(x3, false), true);
        CHECK(std::memcmp(out3->getData().raw(), out->getData().raw(),
                          out3->getData().numel() * sizeof(float)) == 0);
        out3->release_graph();
    }

    auto loss = weighted_sum(out, R);
    loss->backward();

    auto check_grad = [&](float analytic, double numeric) {
        const double err = std::fabs(analytic - numeric);
        const double rel = err / (std::fabs(analytic) + std::fabs(numeric) + 1e-8);
        const bool ok = rel < 3e-2 || err < 2e-3;
        if (!ok) std::fprintf(stderr, "  grad mismatch: analytic %g numeric %g\n", analytic, numeric);
        CHECK(ok);
    };

    const float eps = 1e-2f;
    for (size_t idx : {size_t{0}, size_t{9}, size_t{17}, size_t{S * d - 1}}) {
        Tensor xp = x, xm = x;
        xp.raw()[idx] += eps;
        xm.raw()[idx] -= eps;
        check_grad(input->getGrad().raw()[idx], (loss_at(xp) - loss_at(xm)) / (2.0 * eps));
    }

    // A weight behind the attention dropout (W_v) and one behind the output
    // dropout (W_o).
    for (const auto& W : {attn.getW_v(), attn.getW_o()}) {
        for (size_t idx : {size_t{1}, size_t{27}}) {
            float& w = W->getData().raw()[idx];
            const float saved = w;
            w = saved + eps;
            const double lp = loss_at(x);
            w = saved - eps;
            const double lm = loss_at(x);
            w = saved;
            check_grad(W->getGrad().raw()[idx], (lp - lm) / (2.0 * eps));
        }
    }
}

std::vector<char> read_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

void write_bytes(const std::string& path, const std::vector<char>& bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

template <typename T>
void poke(std::vector<char>& bytes, size_t offset, T value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

// Expects GPTModel::load to throw std::runtime_error whose message
// contains needle.
bool load_fails_with(const std::string& path, const std::string& needle) {
    try {
        (void)GPTModel::load(path);
    } catch (const std::runtime_error& e) {
        const bool ok = std::string(e.what()).find(needle) != std::string::npos;
        if (!ok) std::fprintf(stderr, "  unexpected load error: %s\n", e.what());
        return ok;
    } catch (...) {
        return false;
    }
    return false;
}

void test_checkpoint(GPTArch arch) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() /
        ("grad_core_regressions_" + std::to_string(static_cast<int>(arch)));
    fs::create_directories(dir);
    const std::string good = (dir / "model.bin").string();
    const std::string bad = (dir / "bad.bin").string();

    const int vocab = 11, d = 8, layers = 2, heads = 2, max_len = 6;
    GPTModel model(vocab, d, layers, heads, max_len, 0.0f, arch);
    CHECK(model.save(good, /*quiet=*/true));

    // Round trip: every parameter bitwise equal.
    {
        GPTModel loaded = GPTModel::load(good);
        auto a = model.getAllParameters();
        auto b = loaded.getAllParameters();
        CHECK(a.size() == b.size());
        bool same = a.size() == b.size();
        for (size_t i = 0; same && i < a.size(); i++) {
            const Tensor& x = a[i]->getData();
            const Tensor& y = b[i]->getData();
            same = x.getRows() == y.getRows() && x.getCols() == y.getCols() &&
                   std::memcmp(x.raw(), y.raw(), x.numel() * sizeof(float)) == 0;
        }
        CHECK(same);
    }

    const std::vector<char> bytes = read_bytes(good);
    // Header: magic, version, arch (uint32 each), then vocab, d_model,
    // layers, heads, max_len (int32) and dropout (float32); the token
    // embedding's rows field follows at byte 36.
    CHECK(bytes.size() > 64);

    std::vector<char> truncated(bytes.begin(), bytes.end() - 10);
    write_bytes(bad, truncated);
    CHECK(load_fails_with(bad, "file ends inside final norm beta"));

    std::vector<char> header_only(bytes.begin(), bytes.begin() + 20);
    write_bytes(bad, header_only);
    CHECK(load_fails_with(bad, "file ends while reading num_layers"));

    std::vector<char> b = bytes;
    poke<uint32_t>(b, 8, 7u);
    write_bytes(bad, b);
    CHECK(load_fails_with(bad, "unknown architecture tag 7"));

    b = bytes;
    poke<int32_t>(b, 16, -5);
    write_bytes(bad, b);
    CHECK(load_fails_with(bad, "d_model = -5"));

    b = bytes;
    poke<int32_t>(b, 24, 3);  // 8 % 3 != 0
    write_bytes(bad, b);
    CHECK(load_fails_with(bad, "not divisible"));

    b = bytes;
    poke<int32_t>(b, 36, vocab + 1);
    write_bytes(bad, b);
    CHECK(load_fails_with(bad, "token embedding is 12x8, model expects 11x8"));

    CHECK(load_fails_with(bad, bad));  // the path is in every message

    CHECK(!model.save((dir / "no_such_dir" / "x.bin").string(), true));

    fs::remove_all(dir);
}

}  // namespace

int main() {
    test_dropout_masks();
    test_init_seed();
    test_parallel_for();
    test_attention_2d_dropout_gradients(/*rope=*/false);
    test_attention_2d_dropout_gradients(/*rope=*/true);
    test_checkpoint(GPTArch::GPT2);
    test_checkpoint(GPTArch::Modern);

    if (g_failures) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("All core regression checks passed\n");
    return 0;
}
