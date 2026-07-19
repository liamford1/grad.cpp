#include "transformer/activations.h"
#include "transformer/parallel.h"
#include <cstdint>

void fill_dropout_mask(float* mask, size_t n, float dropout_rate, float scale) {
    // One xorshift128+ draw yields four independent 16-bit lanes, so the
    // RNG steps once per four mask values - per-element draws measured at
    // ~14% of training-step CPU (BENCHMARKS.md #10). 16-bit thresholds
    // quantize the drop rate to 1/65536ths (0.1 becomes 0.100006), far
    // below any effect on training.
    const uint32_t threshold = static_cast<uint32_t>(dropout_rate * 65536.0f);

    // Each pool thread has its own xorshift128+ state, so chunks can fill
    // in parallel without sharing RNG state.
    parallel_for(n, 65536, [&](size_t begin, size_t end) {
        static thread_local uint64_t s0 = 0x9E3779B97F4A7C15ull;
        static thread_local uint64_t s1 = 0xBF58476D1CE4E5B9ull;

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
    });
}

Tensor dropout(const Tensor& input, float dropout_rate, bool training) {
    if (dropout_rate == 0.0f || !training) {
        return input;
    }

    Tensor result = input.getIs3D()
        ? Tensor(input.getBatchSize(), input.getRows(), input.getCols())
        : Tensor(input.getRows(), input.getCols());

    const float scale = 1.0f / (1.0f - dropout_rate);
    const size_t n = input.numel();

    fill_dropout_mask(result.raw(), n, dropout_rate, scale);

    const float* in = input.raw();
    float* out = result.raw();
    for (size_t i = 0; i < n; i++) {
        out[i] *= in[i];
    }
    return result;
}
