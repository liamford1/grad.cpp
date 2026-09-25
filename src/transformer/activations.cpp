#include "transformer/activations.h"
#include "transformer/parallel.h"
#include <atomic>
#include <cstdint>

namespace {

constexpr uint64_t kDefaultDropoutSeed = 0x9E3779B97F4A7C15ull;

// Reseed granularity. Chunk boundaries handed out by parallel_for below are
// multiples of this, and a nested (inline) call walks the same boundaries,
// so the mask never depends on how the range was partitioned.
constexpr size_t kBlock = 65536;

std::atomic<uint64_t> g_seed{kDefaultDropoutSeed};
std::atomic<uint64_t> g_next_stream{0};

// splitmix64 (Steele, Lea, Flood): advances x and returns a well-mixed
// output. Used only to derive xorshift state, never per element.
inline uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

void fill_block(float* mask, size_t begin, size_t end, uint64_t stream_key,
                uint32_t threshold, float scale) {
    // State lives in registers for the block. The block index is folded
    // into the splitmix input, so neighbouring blocks (and neighbouring
    // streams) start from unrelated states.
    uint64_t sm = stream_key ^ (static_cast<uint64_t>(begin / kBlock) * 0xD1B54A32D192ED03ull);
    uint64_t s0 = splitmix64(sm);
    uint64_t s1 = splitmix64(sm);
    if ((s0 | s1) == 0) s1 = 1;  // xorshift128+ must not start at all-zero

    size_t i = begin;
    while (i < end) {
        uint64_t x = s0;
        const uint64_t y = s1;
        s0 = y;
        x ^= x << 23;
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        uint64_t r = s1 + y;

        const size_t lanes = (end - i < 4) ? end - i : 4;
        for (size_t lane = 0; lane < lanes; lane++) {
            mask[i + lane] = ((r & 0xFFFFu) < threshold) ? 0.0f : scale;
            r >>= 16;
        }
        i += lanes;
    }
}

}  // namespace

void set_dropout_seed(uint64_t seed) {
    g_seed.store(seed, std::memory_order_relaxed);
    g_next_stream.store(0, std::memory_order_relaxed);
}

uint64_t reserve_dropout_streams(uint64_t count) {
    return g_next_stream.fetch_add(count, std::memory_order_relaxed);
}

void fill_dropout_mask(float* mask, size_t n, float dropout_rate, float scale) {
    fill_dropout_mask(mask, n, dropout_rate, scale, reserve_dropout_streams(1));
}

void fill_dropout_mask(float* mask, size_t n, float dropout_rate, float scale,
                       uint64_t stream) {
    // One xorshift128+ draw yields four independent 16-bit lanes, so the
    // RNG steps once per four mask values - per-element draws measured at
    // ~14% of training-step CPU (BENCHMARKS.md #10). 16-bit thresholds
    // quantize the drop rate to 1/65536ths (0.1 becomes 0.100006), far
    // below any effect on training.
    const uint32_t threshold = static_cast<uint32_t>(dropout_rate * 65536.0f);

    uint64_t key_state = g_seed.load(std::memory_order_relaxed) ^ stream * 0x9E3779B97F4A7C15ull;
    const uint64_t stream_key = splitmix64(key_state);

    parallel_for(n, kBlock, [&](size_t begin, size_t end) {
        for (size_t b = begin; b < end; b += kBlock) {
            const size_t block_end = (end - b < kBlock) ? end : b + kBlock;
            fill_block(mask, b, block_end, stream_key, threshold, scale);
        }
    });
}
