// Kernels of the resident stream (docs/design/metal-resident.md), compiled
// at runtime from this source, embedded into the library by CMake. ASCII
// only: CMake embeds it byte for byte.
//
// Every parameter struct mirrors one in metal_ops.mm field for field (32-bit
// fields only, so the layouts agree without packing rules). Extents fit in
// 32 bits: a Tensor holds at most 2^30 elements.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------- elementwise

struct EwParams {
    uint n;
    float s;
};

kernel void ew_fill(device float* y [[buffer(0)]], constant EwParams& p [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < p.n) y[i] = p.s;
}

kernel void ew_copy(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                    constant EwParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) y[i] = x[i];
}

kernel void ew_add(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                   device float* y [[buffer(2)]], constant EwParams& p [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
    if (i < p.n) y[i] = a[i] + b[i];
}

// y += x
kernel void ew_acc(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                   constant EwParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) y[i] = y[i] + x[i];
}

// y += x * s. The compiler may fuse this into one rounding where the CPU
// (vDSP scale, then add) rounds twice: a difference of at most an ulp of
// the product.
kernel void ew_acc_scaled(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                          constant EwParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        const float t = x[i] * p.s;
        y[i] = y[i] + t;
    }
}

kernel void ew_mul(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                   device float* y [[buffer(2)]], constant EwParams& p [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
    if (i < p.n) y[i] = a[i] * b[i];
}

// y += a * b
kernel void ew_mul_acc(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                       device float* y [[buffer(2)]], constant EwParams& p [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        const float t = a[i] * b[i];
        y[i] = y[i] + t;
    }
}

kernel void ew_scale(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                     constant EwParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) y[i] = x[i] * p.s;
}

// y *= s[0], a scalar the GPU computed (the gradient clip coefficient).
kernel void ew_scale_by(device const float* s [[buffer(0)]], device float* y [[buffer(1)]],
                        constant EwParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) y[i] = y[i] * s[0];
}

// y[r, c] = x[r, c] + rows[r % period, c] over a (n / cols, cols) matrix: a
// bias row (period 1) or the first `period` rows of a position table added
// to every sequence. x may alias y.
struct RowsParams {
    uint n;
    uint cols;
    uint period;
};

kernel void add_rows(device const float* x [[buffer(0)]], device const float* rows [[buffer(1)]],
                     device float* y [[buffer(2)]], constant RowsParams& p [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    const uint r = i / p.cols;
    const uint c = i % p.cols;
    y[i] = x[i] + rows[(r % p.period) * p.cols + c];
}

// out(batch, R, C) = a + b, each operand read through (batch, row, col)
// strides that are 0 along the axes it broadcasts: Tensor::add's general
// case.
struct BroadcastParams {
    uint batch, rows, cols;
    uint a_batch, a_row, a_col;
    uint b_batch, b_row, b_col;
};

kernel void broadcast_add(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                          device float* y [[buffer(2)]], constant BroadcastParams& p [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.batch * p.rows * p.cols) return;
    const uint n = i / (p.rows * p.cols);
    const uint r = (i / p.cols) % p.rows;
    const uint c = i % p.cols;
    y[i] = a[n * p.a_batch + r * p.a_row + c * p.a_col]
           + b[n * p.b_batch + r * p.b_row + c * p.b_col];
}

// dst(R, C) += the sum of g(batch, rows, cols) over every axis dst was
// broadcast along (the batch always, rows if br, columns if bc), one thread
// per dst element in a fixed order: the general add backward.
struct ReduceParams {
    uint batch, rows, cols;
    uint dst_rows, dst_cols;
    uint br, bc;
};

kernel void broadcast_reduce(device const float* g [[buffer(0)]], device float* dst [[buffer(1)]],
                             constant ReduceParams& p [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= p.dst_rows * p.dst_cols) return;
    const uint oi = i / p.dst_cols;
    const uint oj = i % p.dst_cols;
    float s = 0.0f;
    for (uint n = 0; n < p.batch; n++) {
        const uint r_lo = p.br ? 0 : oi;
        const uint r_hi = p.br ? p.rows : oi + 1;
        for (uint r = r_lo; r < r_hi; r++) {
            const uint c_lo = p.bc ? 0 : oj;
            const uint c_hi = p.bc ? p.cols : oj + 1;
            for (uint c = c_lo; c < c_hi; c++) s += g[(n * p.rows + r) * p.cols + c];
        }
    }
    dst[i] = dst[i] + s;
}

// ------------------------------------------------------------ column sums

// dst[c] += sum over rows of x[r, c] in two fixed-order passes: each
// (column, block of block_rows rows) sums its rows in order, then each
// column sums its block partials in order. Deterministic for any schedule,
// and with ~64 blocks the rows of a (B*S, d) gradient spread over d * 64
// threads instead of d.
struct ColParams {
    uint rows;
    uint cols;
    uint block_rows;
    uint nblocks;
};

kernel void col_partial(device const float* x [[buffer(0)]], device float* part [[buffer(1)]],
                        constant ColParams& p [[buffer(2)]],
                        uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    const uint b = gid.y;
    if (c >= p.cols || b >= p.nblocks) return;
    const uint r0 = b * p.block_rows;
    const uint r1 = min(r0 + p.block_rows, p.rows);
    float s = 0.0f;
    for (uint r = r0; r < r1; r++) s += x[r * p.cols + c];
    part[b * p.cols + c] = s;
}

// dst[c] += sum over b of part[b, c], in order.
kernel void col_finish(device const float* part [[buffer(0)]], device float* dst [[buffer(1)]],
                       constant ColParams& p [[buffer(2)]], uint c [[thread_position_in_grid]]) {
    if (c >= p.cols) return;
    float s = 0.0f;
    for (uint b = 0; b < p.nblocks; b++) s += part[b * p.cols + c];
    dst[c] = dst[c] + s;
}

// ------------------------------------------------------------ activations

// gelu(x) = 0.5x(1 + tanh(k(x + a x^3))), the tanh approximation the CPU
// uses, with the CPU's constants.
constant constexpr float kGeluK = 0.79788456f;
constant constexpr float kGeluA = 0.044715f;

kernel void gelu_fwd(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                     constant EwParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    const float xi = x[i];
    const float t = tanh(kGeluK * (xi + kGeluA * xi * xi * xi));
    y[i] = 0.5f * xi * (1.0f + t);
}

// dx += gelu'(x) * dy
kernel void gelu_bwd(device const float* x [[buffer(0)]], device const float* dy [[buffer(1)]],
                     device float* dx [[buffer(2)]], constant EwParams& p [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    const float xi = x[i];
    const float t = tanh(kGeluK * (xi + kGeluA * xi * xi * xi));
    const float sech_sq = 1.0f - t * t;
    const float d = 0.5f * (1.0f + t + xi * sech_sq * kGeluK * (1.0f + 3.0f * kGeluA * xi * xi));
    dx[i] = dx[i] + d * dy[i];
}

kernel void silu_fwd(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                     constant EwParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    y[i] = x[i] / (1.0f + exp(-x[i]));
}

// dx += dy * s * (1 + x (1 - s)), s = sigmoid(x)
kernel void silu_bwd(device const float* x [[buffer(0)]], device const float* dy [[buffer(1)]],
                     device float* dx [[buffer(2)]], constant EwParams& p [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    const float xi = x[i];
    const float s = 1.0f / (1.0f + exp(-xi));
    dx[i] = dx[i] + dy[i] * s * (1.0f + xi * (1.0f - s));
}

// --------------------------------------------------------- row reductions

// One 256-thread threadgroup per row. Each thread accumulates a fixed
// strided subset of the row, then a fixed tree combines the 256 partials,
// so a row's result never depends on scheduling.
constant constexpr uint kRow = 256;

inline float tg_sum(threadgroup float* buf, float v, uint tid) {
    buf[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = kRow / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float r = buf[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

inline float tg_max(threadgroup float* buf, float v, uint tid) {
    buf[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = kRow / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] = max(buf[tid], buf[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float r = buf[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

struct RowParams {
    uint rows;
    uint cols;
    float eps;    // norms: epsilon
    uint flag;    // norms: RMS; attention backward: has a dropout mask
    float scale;  // attention: 1/sqrt(head_size)
};

// LayerNorm (flag = 0) or RMSNorm (flag = 1) over each row, keeping the
// per-row mean and 1/std (1/rms) for the backward pass. Two passes for the
// variance, like the CPU.
kernel void layer_norm_fwd(device const float* x [[buffer(0)]],
                           device const float* gamma [[buffer(1)]],
                           device const float* beta [[buffer(2)]], device float* y [[buffer(3)]],
                           device float* mean [[buffer(4)]], device float* rstd [[buffer(5)]],
                           constant RowParams& p [[buffer(6)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float buf[kRow];
    const uint d = p.cols;
    const float df = float(d);
    device const float* xr = x + row * d;
    device float* yr = y + row * d;
    if (p.flag) {
        float s = 0.0f;
        for (uint j = tid; j < d; j += kRow) s += xr[j] * xr[j];
        const float r = 1.0f / sqrt(tg_sum(buf, s, tid) / df + p.eps);
        for (uint j = tid; j < d; j += kRow) yr[j] = gamma[j] * xr[j] * r;
        if (tid == 0) {
            mean[row] = 0.0f;
            rstd[row] = r;
        }
        return;
    }
    float s = 0.0f;
    for (uint j = tid; j < d; j += kRow) s += xr[j];
    const float mu = tg_sum(buf, s, tid) / df;
    float v = 0.0f;
    for (uint j = tid; j < d; j += kRow) {
        const float diff = xr[j] - mu;
        v += diff * diff;
    }
    const float inv = 1.0f / sqrt(tg_sum(buf, v, tid) / df + p.eps);
    for (uint j = tid; j < d; j += kRow) yr[j] = gamma[j] * ((xr[j] - mu) * inv) + beta[j];
    if (tid == 0) {
        mean[row] = mu;
        rstd[row] = inv;
    }
}

// dx += the input gradient of each row, with the CPU's formulas (the dvar /
// dmean form for LayerNorm, the direct form for RMSNorm).
kernel void layer_norm_bwd_dx(device const float* x [[buffer(0)]],
                              device const float* gamma [[buffer(1)]],
                              device const float* dy [[buffer(2)]],
                              device const float* mean [[buffer(3)]],
                              device const float* rstd [[buffer(4)]], device float* dx [[buffer(5)]],
                              constant RowParams& p [[buffer(6)]],
                              uint row [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float buf[kRow];
    const uint d = p.cols;
    const float df = float(d);
    device const float* xr = x + row * d;
    device const float* dyr = dy + row * d;
    device float* dxr = dx + row * d;
    const float inv = rstd[row];
    if (p.flag) {
        float dot = 0.0f;
        for (uint j = tid; j < d; j += kRow) dot += dyr[j] * gamma[j] * xr[j];
        const float k = tg_sum(buf, dot, tid) * inv * inv * inv / df;
        for (uint j = tid; j < d; j += kRow) {
            dxr[j] = dxr[j] + (gamma[j] * dyr[j] * inv - xr[j] * k);
        }
        return;
    }
    const float mu = mean[row];
    const float dvar_scale = -0.5f * inv * inv * inv;
    float dv = 0.0f;
    for (uint j = tid; j < d; j += kRow) dv += dyr[j] * gamma[j] * (xr[j] - mu) * dvar_scale;
    const float dvar = tg_sum(buf, dv, tid);
    float dm = 0.0f;
    for (uint j = tid; j < d; j += kRow) {
        dm += dyr[j] * gamma[j] * -inv + dvar * -2.0f * (xr[j] - mu) / df;
    }
    const float dmean = tg_sum(buf, dm, tid);
    for (uint j = tid; j < d; j += kRow) {
        const float xmm = xr[j] - mu;
        dxr[j] = dxr[j] + (dyr[j] * gamma[j] * inv + dvar * 2.0f * xmm / df + dmean / df);
    }
}

// Per-(column, block of rows) partial sums of the gamma and beta gradients,
// finished by col_finish into dgamma and dbeta.
struct NormParamsParams {
    uint rows;
    uint d;
    uint block_rows;
    uint nblocks;
    uint rms;
    uint want_beta;
};

kernel void layer_norm_bwd_params(device const float* x [[buffer(0)]],
                                  device const float* dy [[buffer(1)]],
                                  device const float* mean [[buffer(2)]],
                                  device const float* rstd [[buffer(3)]],
                                  device float* gpart [[buffer(4)]],
                                  device float* bpart [[buffer(5)]],
                                  constant NormParamsParams& p [[buffer(6)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    const uint j = gid.x;
    const uint b = gid.y;
    if (j >= p.d || b >= p.nblocks) return;
    const uint r0 = b * p.block_rows;
    const uint r1 = min(r0 + p.block_rows, p.rows);
    float g = 0.0f;
    float bs = 0.0f;
    for (uint r = r0; r < r1; r++) {
        const float dyv = dy[r * p.d + j];
        const float xv = x[r * p.d + j];
        if (p.rms) {
            g += dyv * xv * rstd[r];
        } else {
            g += dyv * ((xv - mean[r]) * rstd[r]);
            bs += dyv;
        }
    }
    gpart[b * p.d + j] = g;
    if (p.want_beta) bpart[b * p.d + j] = bs;
}

kernel void softmax_fwd(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                        constant RowParams& p [[buffer(2)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float buf[kRow];
    device const float* xr = x + row * p.cols;
    device float* yr = y + row * p.cols;
    float m = -INFINITY;
    for (uint j = tid; j < p.cols; j += kRow) m = max(m, xr[j]);
    m = tg_max(buf, m, tid);
    float s = 0.0f;
    for (uint j = tid; j < p.cols; j += kRow) {
        const float e = exp(xr[j] - m);
        yr[j] = e;
        s += e;
    }
    const float inv = 1.0f / tg_sum(buf, s, tid);
    for (uint j = tid; j < p.cols; j += kRow) yr[j] *= inv;
}

// dx += y * (dy - dot(y, dy)) per row.
kernel void softmax_bwd(device const float* y [[buffer(0)]], device const float* dy [[buffer(1)]],
                        device float* dx [[buffer(2)]], constant RowParams& p [[buffer(3)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float buf[kRow];
    device const float* yr = y + row * p.cols;
    device const float* dyr = dy + row * p.cols;
    device float* dxr = dx + row * p.cols;
    float dot = 0.0f;
    for (uint j = tid; j < p.cols; j += kRow) dot += yr[j] * dyr[j];
    dot = tg_sum(buf, dot, tid);
    for (uint j = tid; j < p.cols; j += kRow) dxr[j] = dxr[j] + yr[j] * (dyr[j] - dot);
}

kernel void log_softmax_fwd(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                            constant RowParams& p [[buffer(2)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float buf[kRow];
    device const float* xr = x + row * p.cols;
    device float* yr = y + row * p.cols;
    float m = -INFINITY;
    for (uint j = tid; j < p.cols; j += kRow) m = max(m, xr[j]);
    m = tg_max(buf, m, tid);
    float s = 0.0f;
    for (uint j = tid; j < p.cols; j += kRow) s += exp(xr[j] - m);
    const float log_sum = log(tg_sum(buf, s, tid));
    for (uint j = tid; j < p.cols; j += kRow) yr[j] = (xr[j] - m) - log_sum;
}

// dx += dy - softmax * sum(dy), softmax = exp(y).
kernel void log_softmax_bwd(device const float* y [[buffer(0)]],
                            device const float* dy [[buffer(1)]], device float* dx [[buffer(2)]],
                            constant RowParams& p [[buffer(3)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float buf[kRow];
    device const float* yr = y + row * p.cols;
    device const float* dyr = dy + row * p.cols;
    device float* dxr = dx + row * p.cols;
    float s = 0.0f;
    for (uint j = tid; j < p.cols; j += kRow) s += dyr[j];
    s = tg_sum(buf, s, tid);
    for (uint j = tid; j < p.cols; j += kRow) dxr[j] = dxr[j] + (dyr[j] - exp(yr[j]) * s);
}

// Causal attention softmax over each (S, S) score row, in place: row i of a
// unit keeps columns j <= i, scaled by 1/sqrt(head_size), and the masked
// columns become exactly 0 (the CPU's -1e9 mask underflows exp to 0).
kernel void attn_softmax_fwd(device float* s [[buffer(0)]], constant RowParams& p [[buffer(1)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float buf[kRow];
    const uint S = p.cols;
    const uint i = row % S;
    device float* sr = s + row * S;
    float m = -INFINITY;
    for (uint j = tid; j <= i; j += kRow) m = max(m, sr[j] * p.scale);
    m = tg_max(buf, m, tid);
    float sum = 0.0f;
    for (uint j = tid; j < S; j += kRow) {
        const float e = j <= i ? exp(sr[j] * p.scale - m) : 0.0f;
        sr[j] = e;
        sum += e;
    }
    const float inv = 1.0f / tg_sum(buf, sum, tid);
    for (uint j = tid; j <= i; j += kRow) sr[j] *= inv;
}

// Attention softmax backward, in place over dp: with dp' = dp * mask
// (through the dropout, when flag is set), ds = P * (dp' - dot(dp', P)) *
// scale.
kernel void attn_softmax_bwd(device const float* P [[buffer(0)]], device float* dp [[buffer(1)]],
                             device const float* mask [[buffer(2)]],
                             constant RowParams& p [[buffer(3)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float buf[kRow];
    const uint S = p.cols;
    const uint i = row % S;
    device const float* pr = P + row * S;
    device float* dr = dp + row * S;
    device const float* mr = mask + row * S;
    float dot = 0.0f;
    for (uint j = tid; j <= i; j += kRow) {
        const float g = p.flag ? dr[j] * mr[j] : dr[j];
        dot += g * pr[j];
    }
    dot = tg_sum(buf, dot, tid);
    for (uint j = tid; j < S; j += kRow) {
        const float g = p.flag ? dr[j] * mr[j] : dr[j];
        dr[j] = j <= i ? pr[j] * (g - dot) * p.scale : 0.0f;
    }
}

// ----------------------------------------------------------------- losses

// A target index outside [0, vocab) contributes nothing, as on the CPU.
inline bool valid_target(float t, uint vocab, thread uint& index) {
    const int ti = int(t);
    index = uint(ti);
    return ti >= 0 && uint(ti) < vocab;
}

// loss[0] = -sum over rows of logp[r, t_r] / rows, one threadgroup: thread
// k sums rows k, k + 256, ... in order, then the fixed tree.
kernel void nll_fwd(device const float* logp [[buffer(0)]], device const float* targets [[buffer(1)]],
                    device float* loss [[buffer(2)]], constant RowParams& p [[buffer(3)]],
                    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float buf[kRow];
    float s = 0.0f;
    for (uint r = tid; r < p.rows; r += kRow) {
        uint t;
        if (valid_target(targets[r], p.cols, t)) s -= logp[r * p.cols + t];
    }
    const float total = tg_sum(buf, s, tid);
    if (tid == 0) loss[0] = total / float(p.rows);
}

// dlogp[r, t_r] += -g / rows
kernel void nll_bwd(device const float* g [[buffer(0)]], device const float* targets [[buffer(1)]],
                    device float* dlogp [[buffer(2)]], constant RowParams& p [[buffer(3)]],
                    uint r [[thread_position_in_grid]]) {
    if (r >= p.rows) return;
    uint t;
    if (valid_target(targets[r], p.cols, t)) {
        const float scale = -g[0] / float(p.rows);
        dlogp[r * p.cols + t] = dlogp[r * p.cols + t] + scale;
    }
}

// Fused log-softmax + NLL forward: per row, stats = (max, log sum exp(x -
// max)) for the backward pass and row_loss = -logp[t], with logp formed
// exactly as log_softmax_fwd forms it.
kernel void ce_fwd(device const float* x [[buffer(0)]], device const float* targets [[buffer(1)]],
                   device float* stats [[buffer(2)]], device float* row_loss [[buffer(3)]],
                   constant RowParams& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float buf[kRow];
    device const float* xr = x + row * p.cols;
    float m = -INFINITY;
    for (uint j = tid; j < p.cols; j += kRow) m = max(m, xr[j]);
    m = tg_max(buf, m, tid);
    float s = 0.0f;
    for (uint j = tid; j < p.cols; j += kRow) s += exp(xr[j] - m);
    const float log_sum = log(tg_sum(buf, s, tid));
    if (tid == 0) {
        stats[2 * row] = m;
        stats[2 * row + 1] = log_sum;
        uint t;
        row_loss[row] = valid_target(targets[row], p.cols, t) ? -((xr[t] - m) - log_sum) : 0.0f;
    }
}

// out[0] = sum(x) / n, one threadgroup, fixed order.
kernel void mean_reduce(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                        constant EwParams& p [[buffer(2)]],
                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float buf[kRow];
    float s = 0.0f;
    for (uint i = tid; i < p.n; i += kRow) s += x[i];
    const float total = tg_sum(buf, s, tid);
    if (tid == 0) out[0] = total / float(p.n);
}

// Fused backward: dx += dy - softmax * sum(dy) with the NLL gradient dy
// (-g / rows at the target, 0 elsewhere) formed in place, never stored.
kernel void ce_bwd(device const float* x [[buffer(0)]], device const float* stats [[buffer(1)]],
                   device const float* targets [[buffer(2)]], device const float* g [[buffer(3)]],
                   device float* dx [[buffer(4)]], constant RowParams& p [[buffer(5)]],
                   uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.cols) return;
    const uint r = i / p.cols;
    const uint j = i % p.cols;
    uint t;
    const bool valid = valid_target(targets[r], p.cols, t);
    const float scale = -g[0] / float(p.rows);
    const float soft = exp((x[i] - stats[2 * r]) - stats[2 * r + 1]);
    const float dy = (valid && j == t) ? scale : 0.0f;
    const float sum = valid ? scale : 0.0f;
    dx[i] = dx[i] + (dy - soft * sum);
}

// ------------------------------------------------------------- embeddings

struct EmbedParams {
    uint n;  // forward: tokens * d; backward: unique tokens * d
    uint d;
    float scale;
};

kernel void embedding_fwd(device const float* ids [[buffer(0)]],
                          device const float* table [[buffer(1)]], device float* out [[buffer(2)]],
                          constant EmbedParams& p [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    const uint t = i / p.d;
    const uint j = i % p.d;
    out[i] = table[uint(int(ids[t])) * p.d + j] * p.scale;
}

// Segmented scatter-add, one thread per (distinct token, column): the
// token's positions, listed in increasing order by the CPU's stable
// counting sort, are summed in that order, the order the CPU loop visits
// them. No two threads write the same element, so no atomics.
kernel void embedding_bwd(device const uint* tokens [[buffer(0)]],
                          device const uint* starts [[buffer(1)]],
                          device const uint* order [[buffer(2)]],
                          device const float* dout [[buffer(3)]], device float* dtable [[buffer(4)]],
                          constant EmbedParams& p [[buffer(5)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    const uint u = i / p.d;
    const uint j = i % p.d;
    device float* dst = dtable + tokens[u] * p.d + j;
    float acc = *dst;
    for (uint k = starts[u]; k < starts[u + 1]; k++) {
        const float t = dout[order[k] * p.d + j] * p.scale;
        acc = acc + t;
    }
    *dst = acc;
}

// Rotates each head's (2j, 2j+1) pairs of a (rows, d) block by position
// (row % S) angles from the CPU-built tables; inverse applies the
// transpose rotation (the backward pass).
struct RopeParams {
    uint rows;
    uint S;
    uint d;
    uint half_head;
    uint inverse;
};

kernel void rope(device float* x [[buffer(0)]], device const float* cos_t [[buffer(1)]],
                 device const float* sin_t [[buffer(2)]], constant RopeParams& p [[buffer(3)]],
                 uint i [[thread_position_in_grid]]) {
    const uint pairs = p.d / 2;
    if (i >= p.rows * pairs) return;
    const uint row = i / pairs;
    const uint pair = i % pairs;
    const uint h = pair / p.half_head;
    const uint j = pair % p.half_head;
    const uint pos = row % p.S;
    const float c = cos_t[pos * p.half_head + j];
    const float s = p.inverse ? -sin_t[pos * p.half_head + j] : sin_t[pos * p.half_head + j];
    device float* head = x + row * p.d + h * 2 * p.half_head;
    const float a = head[2 * j];
    const float b = head[2 * j + 1];
    head[2 * j] = a * c - b * s;
    head[2 * j + 1] = a * s + b * c;
}

// ---------------------------------------------------------------- dropout

// The CPU mask generator, reproduced bit for bit (activations.cpp): unit u
// uses stream first_stream + u; each 65,536-element block of it reseeds
// xorshift128+ through splitmix64 from (seed, stream, block), and each step
// yields four 16-bit lanes compared against the threshold. The generator is
// linear over GF(2), so the block's 16,384 steps split across 256 threads:
// thread k starts from the block state times jump[k] = T^(64k) (built on
// the CPU; 128 rows of 128 bits) and takes 64 steps.
struct DropoutParams {
    ulong seed;
    ulong first_stream;
    uint n;  // elements per unit
    uint units;
    uint blocks;  // 65,536-element blocks per unit
    uint threshold;
    float scale;
    uint has_x;
};

constant constexpr uint kDropBlock = 65536;
constant constexpr uint kDropSteps = 64;  // steps per thread; 4 lanes each

inline ulong splitmix64(thread ulong& x) {
    ulong z = (x += 0x9E3779B97F4A7C15ul);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
    return z ^ (z >> 31);
}

kernel void dropout_fwd(device const float* x [[buffer(0)]], device float* mask [[buffer(1)]],
                        device float* y [[buffer(2)]], device const ulong* jump [[buffer(3)]],
                        constant DropoutParams& p [[buffer(4)]],
                        uint gid [[thread_position_in_grid]]) {
    const uint lane = gid % 256;
    const uint block_global = gid / 256;
    const uint unit = block_global / p.blocks;
    const uint block = block_global % p.blocks;
    if (unit >= p.units) return;
    const uint begin = block * kDropBlock + lane * kDropSteps * 4;
    if (begin >= p.n) return;
    const uint end = min(begin + kDropSteps * 4, p.n);

    ulong key_state = p.seed ^ ((p.first_stream + unit) * 0x9E3779B97F4A7C15ul);
    const ulong stream_key = splitmix64(key_state);
    ulong sm = stream_key ^ (ulong(block) * 0xD1B54A32D192ED03ul);
    ulong s0 = splitmix64(sm);
    ulong s1 = splitmix64(sm);
    if ((s0 | s1) == 0) s1 = 1;

    device const ulong* rows = jump + lane * 256;
    ulong n0 = 0;
    ulong n1 = 0;
    for (uint r = 0; r < 64; r++) {
        const ulong bit = (popcount(rows[2 * r] & s0) ^ popcount(rows[2 * r + 1] & s1)) & 1ul;
        n0 |= bit << r;
    }
    for (uint r = 0; r < 64; r++) {
        const ulong bit =
            (popcount(rows[128 + 2 * r] & s0) ^ popcount(rows[128 + 2 * r + 1] & s1)) & 1ul;
        n1 |= bit << r;
    }
    s0 = n0;
    s1 = n1;

    const uint base = unit * p.n;
    for (uint i = begin; i < end;) {
        ulong xs = s0;
        const ulong ys = s1;
        s0 = ys;
        xs ^= xs << 23;
        s1 = xs ^ ys ^ (xs >> 17) ^ (ys >> 26);
        ulong r = s1 + ys;
        const uint lanes = min(4u, end - i);
        for (uint l = 0; l < lanes; l++) {
            const float m = (uint(r & 0xFFFFul) < p.threshold) ? 0.0f : p.scale;
            mask[base + i + l] = m;
            if (p.has_x) y[base + i + l] = x[base + i + l] * m;
            r >>= 16;
        }
        i += lanes;
    }
}

// -------------------------------------------------------------- optimizer

struct AdamParams {
    uint n;
    float b1, b2;
    float inv_bc1, inv_bc2;
    float lr, eps, wd;
};

// AdamW with the CPU's update, operation for operation.
kernel void adamw(device float* w [[buffer(0)]], device const float* g [[buffer(1)]],
                  device float* m [[buffer(2)]], device float* v [[buffer(3)]],
                  constant AdamParams& p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    const float gi = g[i];
    const float mi = p.b1 * m[i] + (1.0f - p.b1) * gi;
    const float vi = p.b2 * v[i] + (1.0f - p.b2) * (gi * gi);
    m[i] = mi;
    v[i] = vi;
    const float m_hat = mi * p.inv_bc1;
    const float v_hat = vi * p.inv_bc2;
    w[i] -= p.lr * (m_hat / (sqrt(v_hat) + p.eps) + p.wd * w[i]);
}

// Gradient norm: each threadgroup sums the squares of a fixed 4,096-element
// chunk (thread k takes elements k, k + 256, ... in order, then the tree)
// into part[offset + chunk]; norm_finish sums every partial in order.
struct SumsqParams {
    uint n;
    uint offset;
};

constant constexpr uint kNormChunk = 4096;

kernel void sumsq_partial(device const float* x [[buffer(0)]], device float* part [[buffer(1)]],
                          constant SumsqParams& p [[buffer(2)]],
                          uint chunk [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float buf[kRow];
    const uint base = chunk * kNormChunk;
    float s = 0.0f;
    for (uint k = tid; k < kNormChunk; k += kRow) {
        const uint idx = base + k;
        if (idx < p.n) s += x[idx] * x[idx];
    }
    const float total = tg_sum(buf, s, tid);
    if (tid == 0) part[p.offset + chunk] = total;
}

// out[0] = sqrt(sum of partials); out[1] = the clip coefficient,
// max_norm / (norm + 1e-6) when norm > max_norm, else 1 (the CPU's rule).
struct NormParams {
    uint count;
    float max_norm;
};

kernel void norm_finish(device const float* part [[buffer(0)]], device float* out [[buffer(1)]],
                        constant NormParams& p [[buffer(2)]],
                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float buf[kRow];
    float s = 0.0f;
    for (uint i = tid; i < p.count; i += kRow) s += part[i];
    const float total = sqrt(tg_sum(buf, s, tid));
    if (tid == 0) {
        out[0] = total;
        out[1] = total > p.max_norm ? p.max_norm / (total + 1e-6f) : 1.0f;
    }
}

// ---------------------------------------------------------------- batched GEMM

// C[z] = alpha * op(A[z]) @ op(B[z]) + beta * C[z] for z in [0, batch), where
// matrix z of each operand starts at (z / inner) * outer + (z % inner) *
// inner_stride. A threadgroup of four simdgroups computes a 32x32 tile of C,
// each simdgroup a 16x16 quarter as 2x2 8x8 simdgroup matrices, stepping K
// 16 at a time through threadgroup memory. The K loop order is fixed, so
// results do not depend on scheduling.
struct GemmParams {
    uint M, N, K;
    uint lda, ldb, ldc;
    uint transA, transB;
    float alpha, beta;
    uint inner;
    uint a_outer, a_inner, b_outer, b_inner, c_outer, c_inner;
};

constant constexpr uint kTileM = 32;
constant constexpr uint kTileN = 32;
constant constexpr uint kTileK = 16;
constant constexpr uint kGemmThreads = 128;

kernel void gemm_batched(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                         device float* C [[buffer(2)]], constant GemmParams& p [[buffer(3)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float As[kTileM * kTileK];
    threadgroup float Bs[kTileK * kTileN];
    threadgroup float Cs[kTileM * kTileN];

    const uint zo = group.z / p.inner;
    const uint zi = group.z % p.inner;
    A += zo * p.a_outer + zi * p.a_inner;
    B += zo * p.b_outer + zi * p.b_inner;
    C += zo * p.c_outer + zi * p.c_inner;

    const uint m0 = group.y * kTileM;
    const uint n0 = group.x * kTileN;
    const uint sm = (sg / 2) * 16;
    const uint sn = (sg % 2) * 16;

    simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < p.K; k0 += kTileK) {
        for (uint e = tid; e < kTileM * kTileK; e += kGemmThreads) {
            const uint i = e / kTileK;
            const uint k = e % kTileK;
            const uint gm = m0 + i;
            const uint gk = k0 + k;
            float v = 0.0f;
            if (gm < p.M && gk < p.K) v = p.transA ? A[gk * p.lda + gm] : A[gm * p.lda + gk];
            As[e] = v;
        }
        for (uint e = tid; e < kTileK * kTileN; e += kGemmThreads) {
            const uint k = e / kTileN;
            const uint j = e % kTileN;
            const uint gk = k0 + k;
            const uint gn = n0 + j;
            float v = 0.0f;
            if (gk < p.K && gn < p.N) v = p.transB ? B[gn * p.ldb + gk] : B[gk * p.ldb + gn];
            Bs[e] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < kTileK; kk += 8) {
            simdgroup_float8x8 a0, a1, b0, b1;
            simdgroup_load(a0, As + (sm + 0) * kTileK + kk, kTileK);
            simdgroup_load(a1, As + (sm + 8) * kTileK + kk, kTileK);
            simdgroup_load(b0, Bs + kk * kTileN + sn + 0, kTileN);
            simdgroup_load(b1, Bs + kk * kTileN + sn + 8, kTileN);
            simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
            simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
            simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
            simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc00, Cs + (sm + 0) * kTileN + sn + 0, kTileN);
    simdgroup_store(acc01, Cs + (sm + 0) * kTileN + sn + 8, kTileN);
    simdgroup_store(acc10, Cs + (sm + 8) * kTileN + sn + 0, kTileN);
    simdgroup_store(acc11, Cs + (sm + 8) * kTileN + sn + 8, kTileN);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = tid; e < kTileM * kTileN; e += kGemmThreads) {
        const uint gm = m0 + e / kTileN;
        const uint gn = n0 + e % kTileN;
        if (gm < p.M && gn < p.N) {
            float r = p.alpha * Cs[e];
            // beta = 0 must not read C: it may be uninitialized, and 0 * NaN
            // is NaN.
            if (p.beta != 0.0f) r += p.beta * C[gm * p.ldc + gn];
            C[gm * p.ldc + gn] = r;
        }
    }
}
