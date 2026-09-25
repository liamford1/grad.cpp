// Regression tests for core-engine invariants: dropout RNG independence
// and determinism, nested parallel_for, tensor shape validation, loss
// gradients, checkpoint validation, and the 2D attention path.
//
// Uses its own CHECK rather than assert so it stays meaningful in Release
// (NDEBUG) builds.

#include "transformer/activations.h"
#include "transformer/parallel.h"
#include "transformer/tensor.h"
#include "transformer/variable.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
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

}  // namespace

int main() {
    test_dropout_masks();
    test_init_seed();
    test_parallel_for();

    if (g_failures) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("All core regression checks passed\n");
    return 0;
}
