#pragma once

#include <cstddef>
#include <cstdint>

// Fills mask[i] with 0 (dropped) or scale (kept), keeping each element with
// probability 1 - dropout_rate. Draws come from xorshift128+ compared in the
// integer domain, a few cycles per element instead of a std::mt19937 +
// uniform_real_distribution draw.
//
// Every mask is a pure function of (dropout seed, stream index, position):
// the generator is reseeded through splitmix64 from those three at each
// 65536-element block, so masks are independent across calls and threads,
// and do not depend on how the thread pool happened to split the range.
//
// The first overload takes the next stream index from a process-wide atomic
// counter. Callers that fill several masks concurrently (attention's
// per-(batch, head) units) reserve a block of indices up front and pass one
// per mask, so the assignment is independent of thread scheduling.
void fill_dropout_mask(float* mask, size_t n, float dropout_rate, float scale);
void fill_dropout_mask(float* mask, size_t n, float dropout_rate, float scale,
                       uint64_t stream);

// Reserves count consecutive stream indices and returns the first.
[[nodiscard]] uint64_t reserve_dropout_streams(uint64_t count);

// Sets the dropout seed and restarts the stream counter, so the masks that
// follow are reproducible. The default seed is fixed; a run that wants
// fresh masks per process must seed explicitly.
void set_dropout_seed(uint64_t seed);
