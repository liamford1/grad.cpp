#include "transformer/activations.h"
#include <cstdint>

void fill_dropout_mask(float* mask, size_t n, float dropout_rate, float scale) {
    // xorshift128+ - fast, and plenty for dropout masks.
    static thread_local uint64_t s0 = 0x9E3779B97F4A7C15ull;
    static thread_local uint64_t s1 = 0xBF58476D1CE4E5B9ull;

    const uint32_t threshold =
        static_cast<uint32_t>(dropout_rate * 4294967296.0);

    for (size_t i = 0; i < n; i++) {
        uint64_t x = s0;
        const uint64_t y = s1;
        s0 = y;
        x ^= x << 23;
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        const uint32_t r = static_cast<uint32_t>(s1 + y);
        mask[i] = (r < threshold) ? 0.0f : scale;
    }
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
