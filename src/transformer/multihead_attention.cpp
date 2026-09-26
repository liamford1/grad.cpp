#include "grad/transformer/tensor.h"
#include "grad/transformer/activations.h"
#include "grad/transformer/multihead_attention.h"
#include <cmath>
#include <stdexcept>
#include <cstring>
#include "grad/transformer/blas_wrapper.h"
#include "grad/transformer/device.h"
#include "grad/transformer/metal_ops.h"
#include "grad/transformer/parallel.h"
#include "metal_graph.h"
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace grad {

namespace {

inline void fill_bias_rows(float* out, const float* bias, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        std::memcpy(out + i * cols, bias, cols * sizeof(float));
    }
}

inline void add_column_sums(const float* x, size_t rows, size_t cols, float* out) {
    for (size_t i = 0; i < rows; i++) {
        const float* row = x + i * cols;
        for (size_t j = 0; j < cols; j++) {
            out[j] += row[j];
        }
    }
}

// cos/sin tables for RoPE: row i holds cos/sin(i * theta_j) for the
// head_size/2 rotation frequencies theta_j = 10000^(-2j/head_size).
void build_rope_tables(size_t seq_len, size_t head_size, std::vector<float>& cos_t,
                       std::vector<float>& sin_t) {
    const size_t half = head_size / 2;
    cos_t.resize(seq_len * half);
    sin_t.resize(seq_len * half);
    for (size_t j = 0; j < half; j++) {
        const float theta =
            std::pow(10000.0f, -2.0f * static_cast<float>(j) / static_cast<float>(head_size));
        for (size_t i = 0; i < seq_len; i++) {
            cos_t[i * half + j] = std::cos(static_cast<float>(i) * theta);
            sin_t[i * half + j] = std::sin(static_cast<float>(i) * theta);
        }
    }
}

// Rotates each head's (2j, 2j+1) pairs in a (seq_len, d_model) block by
// its row's position angle, in place. inverse applies the transpose
// rotation - backward through RoPE, since rotations are orthogonal.
void rope_apply(float* buf, size_t seq_len, size_t d_model, size_t num_heads, size_t head_size,
                const float* cos_t, const float* sin_t, bool inverse) {
    const size_t half = head_size / 2;
    for (size_t i = 0; i < seq_len; i++) {
        const float* c_row = cos_t + i * half;
        const float* s_row = sin_t + i * half;
        float* row = buf + i * d_model;
        for (size_t h = 0; h < num_heads; h++) {
            float* head = row + h * head_size;
            for (size_t j = 0; j < half; j++) {
                const float c = c_row[j];
                const float s = inverse ? -s_row[j] : s_row[j];
                const float a = head[2 * j];
                const float b = head[2 * j + 1];
                head[2 * j] = a * c - b * s;
                head[2 * j + 1] = a * s + b * c;
            }
        }
    }
}

// Same values in a new shape (equal numel, both contiguous row-major), with
// the gradient passed straight back. Used to run 2D input through the
// batched attention path.
std::shared_ptr<Variable> relayout(const std::shared_ptr<Variable>& src, Tensor&& shaped) {
    const bool metal = metal_mode();
    if (metal) {
        metal::ops::copy(src->getData().device_data(), shaped.device_data(), shaped.numel());
    } else {
        std::memcpy(shaped.raw(), src->getData().raw(), shaped.numel() * sizeof(float));
    }
    const bool needs_grad = compute_requires_grad(src);
    auto out = Variable::create(std::move(shaped), needs_grad);
    if (needs_grad) {
        out->setBackward({src}, [src, metal](Variable& node) {
            const Tensor& g = node.getGrad();
            if (metal) {
                metal::ops::accumulate(metal_graph::grad_for_write(*src), g.device_data(),
                                       g.numel());
                return;
            }
            src->ensureGrad();
            float* dst = src->getGrad().raw();
            blas_vadd(dst, g.raw(), dst, g.numel());
        });
    }
    return out;
}

// RoPE tables for Metal mode, built once per (seq_len, head_size) by the
// CPU builder above and kept on the GPU; rebuilding them per layer per
// step would put S * head_size trigonometric calls on the encoding thread.
struct RopeTables {
    Tensor cos_t;
    Tensor sin_t;
};

const RopeTables& metal_rope_tables(size_t seq_len, size_t head_size) {
    static std::mutex mu;
    static std::map<std::pair<size_t, size_t>, std::unique_ptr<RopeTables>> cache;
    std::lock_guard<std::mutex> lk(mu);
    auto& entry = cache[{seq_len, head_size}];
    if (!entry) {
        std::vector<float> cos_v;
        std::vector<float> sin_v;
        build_rope_tables(seq_len, head_size, cos_v, sin_v);
        entry = std::make_unique<RopeTables>();
        entry->cos_t = Tensor::uninitialized(Shape{cos_v.size()});
        entry->sin_t = Tensor::uninitialized(Shape{sin_v.size()});
        // Fresh tensors, never given to the GPU yet: these writes do not wait.
        std::memcpy(entry->cos_t.raw(), cos_v.data(), cos_v.size() * sizeof(float));
        std::memcpy(entry->sin_t.raw(), sin_v.data(), sin_v.size() * sizeof(float));
    }
    return *entry;
}

struct AttentionParams {
    std::shared_ptr<Variable> input;
    std::shared_ptr<Variable> Wq, Wk, Wv, Wo, bq, bk, bv, bo;
    size_t batch, S, d, H, head_size;
    float dropout_rate;
    bool use_dropout;
    bool rope;
    bool needs_grad;
};

// Metal mode: the 3D attention path of forward() below, stage for stage.
// Projections are MPS GEMMs over (B*S, d); the per-(batch, head) products
// are one batched strided GEMM each, reading heads as column slices exactly
// like the CPU's strided BLAS calls; the softmax output (and dropout mask)
// are kept for backward when there is one.
std::shared_ptr<Variable> attention_metal(const AttentionParams& a) {
    namespace ops = metal::ops;
    const size_t B = a.batch;
    const size_t S = a.S;
    const size_t d = a.d;
    const size_t H = a.H;
    const size_t hs = a.head_size;
    const size_t flat = B * S;
    const size_t units = B * H;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hs));
    const float keep_scale = 1.0f / (1.0f - a.dropout_rate);
    const float* x = a.input->getData().device_data();

    auto Q = std::make_shared<Tensor>(Tensor::uninitialized(flat, d));
    auto K = std::make_shared<Tensor>(Tensor::uninitialized(flat, d));
    auto V = std::make_shared<Tensor>(Tensor::uninitialized(flat, d));
    auto concat = std::make_shared<Tensor>(Tensor::uninitialized(flat, d));
    const auto project = [&](const std::shared_ptr<Variable>& W, const std::shared_ptr<Variable>& b,
                             Tensor& out) {
        ops::gemm(x, W->getData().device_data(), out.device_data(), flat, d, d, false, false, 1.0f,
                  0.0f);
        ops::add_rows(out.device_data(), b->getData().device_data(), out.device_data(), flat * d, d,
                      1);
    };
    project(a.Wq, a.bq, *Q);
    project(a.Wk, a.bk, *K);
    project(a.Wv, a.bv, *V);

    const RopeTables* rope = a.rope ? &metal_rope_tables(S, hs) : nullptr;
    if (rope) {
        ops::rope(Q->device_data(), flat, S, d, hs, rope->cos_t.device_data(),
                  rope->sin_t.device_data(), false);
        ops::rope(K->device_data(), flat, S, d, hs, rope->cos_t.device_data(),
                  rope->sin_t.device_data(), false);
    }

    // Head operands of a (B*S, d) projection: z = b * H + h starts at
    // b * S * d + h * hs, rows d apart. Per-unit (S, S) matrices are
    // contiguous, S * S apart. Captures by value: the backward closure
    // keeps a copy after this frame is gone.
    const auto head_gemm = [units, H, S, d, hs](const float* A, bool a_heads, const float* Bm,
                                                bool b_heads, float* C, bool c_heads, size_t M,
                                                size_t N, size_t Kd, bool tA, bool tB) {
        ops::BatchedGemm g;
        g.A = A, g.B = Bm, g.C = C;
        g.M = M, g.N = N, g.K = Kd;
        g.transA = tA, g.transB = tB;
        g.batch = units, g.inner = H;
        g.lda = a_heads ? d : S, g.a_outer = a_heads ? S * d : H * S * S;
        g.a_inner = a_heads ? hs : S * S;
        g.ldb = b_heads ? d : S, g.b_outer = b_heads ? S * d : H * S * S;
        g.b_inner = b_heads ? hs : S * S;
        g.ldc = c_heads ? d : S, g.c_outer = c_heads ? S * d : H * S * S;
        g.c_inner = c_heads ? hs : S * S;
        ops::gemm_batched(g);
    };

    // scores = Q K^T, softmax in place: P.
    auto P = std::make_shared<Tensor>(Tensor::uninitialized(units, S, S));
    head_gemm(Q->device_data(), true, K->device_data(), true, P->device_data(), false, S, S, hs,
              false, true);
    ops::attention_softmax(P->device_data(), units * S, S, scale);

    std::shared_ptr<Tensor> mask;
    Tensor dropped;
    const float* weights = P->device_data();
    if (a.use_dropout) {
        mask = std::make_shared<Tensor>(Tensor::uninitialized(units, S, S));
        dropped = Tensor::uninitialized(units, S, S);
        // One stream per (batch, head) unit, reserved as the CPU path does.
        const uint64_t stream_base = reserve_dropout_streams(static_cast<uint64_t>(units));
        ops::dropout(P->device_data(), mask->device_data(), dropped.device_data(), S * S, units,
                     current_dropout_seed(), stream_base, a.dropout_rate, keep_scale);
        weights = dropped.device_data();
    }
    head_gemm(weights, false, V->device_data(), true, concat->device_data(), true, S, hs, S, false,
              false);

    Tensor out = Tensor::uninitialized(B, S, d);
    ops::gemm(concat->device_data(), a.Wo->getData().device_data(), out.device_data(), flat, d, d,
              false, false, 1.0f, 0.0f);
    ops::add_rows(out.device_data(), a.bo->getData().device_data(), out.device_data(), flat * d, d,
                  1);

    auto output = Variable::create(std::move(out), a.needs_grad);
    if (!a.needs_grad) return output;

    const bool dropout_active = a.use_dropout;
    output->setBackward({a.input, a.Wq, a.Wk, a.Wv, a.Wo, a.bq, a.bk, a.bv, a.bo},
                        [a, Q, K, V, concat, P, mask, rope, head_gemm, flat, units, S, d, scale,
                         dropout_active](Variable& node) {
        namespace ops = metal::ops;
        using metal_graph::grad_for_write;
        const float* dOut = node.getGrad().device_data();

        // Output projection: dWo += concat^T dOut ; dbo += column sums.
        if (a.Wo->requiresGrad()) {
            ops::gemm(concat->device_data(), dOut, grad_for_write(*a.Wo), d, d, flat, true, false,
                      1.0f, 1.0f);
        }
        if (a.bo->requiresGrad()) ops::column_sums(dOut, flat, d, grad_for_write(*a.bo));

        const bool grad_input = a.input->requiresGrad();
        if (!grad_input && !a.Wq->requiresGrad() && !a.Wk->requiresGrad() && !a.Wv->requiresGrad()
            && !a.bq->requiresGrad() && !a.bk->requiresGrad() && !a.bv->requiresGrad()) {
            return;
        }

        Tensor dConcat = Tensor::uninitialized(flat, d);
        ops::gemm(dOut, a.Wo->getData().device_data(), dConcat.device_data(), flat, d, d, false,
                  true, 1.0f, 0.0f);

        // The weights the forward multiplied V by: P, or P * mask.
        Tensor effective;
        const float* w_eff = P->device_data();
        if (dropout_active) {
            effective = Tensor::uninitialized(units, S, S);
            ops::mul(P->device_data(), mask->device_data(), effective.device_data(), units * S * S);
            w_eff = effective.device_data();
        }

        // attended = W_eff V: dW_eff = dAttended V^T ; dV = W_eff^T dAttended.
        Tensor dP = Tensor::uninitialized(units, S, S);
        Tensor dQ = Tensor::uninitialized(flat, d);
        Tensor dK = Tensor::uninitialized(flat, d);
        Tensor dV = Tensor::uninitialized(flat, d);
        head_gemm(dConcat.device_data(), true, V->device_data(), true, dP.device_data(), false, S,
                  S, a.head_size, false, true);
        head_gemm(w_eff, false, dConcat.device_data(), true, dV.device_data(), true, S, a.head_size,
                  S, true, false);
        // Through the dropout and the softmax, in place: dP becomes dScores.
        ops::attention_softmax_backward(P->device_data(), dP.device_data(),
                                        dropout_active ? mask->device_data() : nullptr, units * S,
                                        S, scale);
        // dQ = dS K ; dK = dS^T Q
        head_gemm(dP.device_data(), false, K->device_data(), true, dQ.device_data(), true, S,
                  a.head_size, S, false, false);
        head_gemm(dP.device_data(), false, Q->device_data(), true, dK.device_data(), true, S,
                  a.head_size, S, true, false);

        // Gradients w.r.t. the rotated Q and K, rotated back.
        if (rope) {
            ops::rope(dQ.device_data(), flat, S, d, a.head_size, rope->cos_t.device_data(),
                      rope->sin_t.device_data(), true);
            ops::rope(dK.device_data(), flat, S, d, a.head_size, rope->cos_t.device_data(),
                      rope->sin_t.device_data(), true);
        }

        const float* x_in = a.input->getData().device_data();
        const auto weight_grad = [&](const std::shared_ptr<Variable>& W, const Tensor& dProj) {
            if (!W->requiresGrad()) return;
            ops::gemm(x_in, dProj.device_data(), grad_for_write(*W), d, d, flat, true, false, 1.0f,
                      1.0f);
        };
        const auto bias_grad = [&](const std::shared_ptr<Variable>& b, const Tensor& dProj) {
            if (!b->requiresGrad()) return;
            ops::column_sums(dProj.device_data(), flat, d, grad_for_write(*b));
        };
        weight_grad(a.Wq, dQ);
        weight_grad(a.Wk, dK);
        weight_grad(a.Wv, dV);
        bias_grad(a.bq, dQ);
        bias_grad(a.bk, dK);
        bias_grad(a.bv, dV);

        if (grad_input) {
            float* dIn = grad_for_write(*a.input);
            ops::gemm(dQ.device_data(), a.Wq->getData().device_data(), dIn, flat, d, d, false, true,
                      1.0f, 1.0f);
            ops::gemm(dK.device_data(), a.Wk->getData().device_data(), dIn, flat, d, d, false, true,
                      1.0f, 1.0f);
            ops::gemm(dV.device_data(), a.Wv->getData().device_data(), dIn, flat, d, d, false, true,
                      1.0f, 1.0f);
        }
    });
    return output;
}

}  // namespace

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads, float dropout_rate, bool rope)
    : d_model_(d_model), num_heads_(num_heads), dropout_rate_(dropout_rate), rope_(rope) {
    if (d_model <= 0 || num_heads <= 0 || (d_model % num_heads) != 0) {
        throw std::invalid_argument("d_model must be divisible by num_heads and num_heads > 0");
    }
    if (rope && (d_model / num_heads) % 2 != 0) {
        throw std::invalid_argument("RoPE requires an even head size");
    }

    const size_t d = static_cast<size_t>(d_model);
    Tensor wq_tensor(d, d);
    Tensor wk_tensor(d, d);
    Tensor wv_tensor(d, d);
    Tensor wo_tensor(d, d);

    wq_tensor.xavier(d, d);
    wk_tensor.xavier(d, d);
    wv_tensor.xavier(d, d);
    wo_tensor.xavier(d, d);

    W_q = Variable::create(wq_tensor, true);
    W_k = Variable::create(wk_tensor, true);
    W_v = Variable::create(wv_tensor, true);
    W_o = Variable::create(wo_tensor, true);

    // Biases start at the constructor's zeros.
    b_q = Variable::create(Tensor(1, d), true);
    b_k = Variable::create(Tensor(1, d), true);
    b_v = Variable::create(Tensor(1, d), true);
    b_o = Variable::create(Tensor(1, d), true);
}

std::shared_ptr<Variable> MultiHeadAttention::forward(const std::shared_ptr<Variable>& input,
                                                      bool training) const {
    const Tensor& input_tensor = input->getData();

    if (input_tensor.getCols() != static_cast<size_t>(d_model_)) {
        throw std::invalid_argument("MultiHeadAttention: input width "
                                    + std::to_string(input_tensor.getCols())
                                    + " does not match d_model " + std::to_string(d_model_));
    }

    if (!input_tensor.getIs3D()) {
        // A 2D (seq, d_model) input is a single sequence: run it through the
        // batched path as batch 1 and hand back a 2D result, so there is one
        // forward/backward implementation (dropout masks, RoPE and all) to
        // keep correct. Both relayouts are one memcpy of the activations;
        // training and generation never take this path.
        const size_t S = input_tensor.getRows();
        auto batched =
            forward(relayout(input, Tensor::uninitialized(1, S, input_tensor.getCols())), training);
        return relayout(batched, Tensor::uninitialized(S, batched->getData().getCols()));
    }

    // (batch, seq, d_model): the training path.
    //
    // Layout notes: a 3D tensor is contiguous, so all four projections
    // run as single (batch*seq, d) sgemms. Attention itself is
    // batch*heads independent (seq, head_size) problems, executed in
    // parallel with per-task scratch. When the graph is recorded, the
    // softmax output (and the attention-dropout mask, when active) are
    // cached for the backward pass, so it recomputes nothing and
    // differentiates through exactly the mask the forward applied; without
    // it they live only in per-task scratch.
    // Extents come from the input's shape; num_heads_ was checked positive
    // at construction.
    const size_t batch_size = input_tensor.getBatchSize();
    const size_t S = input_tensor.getRows();
    const size_t d = input_tensor.getCols();
    const size_t H = static_cast<size_t>(num_heads_);
    const size_t head_size = d / H;
    const size_t flat = batch_size * S;

    const bool use_attn_dropout = training && dropout_rate_ > 0.0f;
    // The weights get gradients even when the input is frozen, so the
    // graph is recorded if anything feeding this op requires grad.
    const bool needs_grad = compute_requires_grad(input, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o);
    if (metal_mode()) {
        auto output =
            attention_metal({input, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, batch_size, S, d, H,
                             head_size, dropout_rate_, use_attn_dropout, rope_, needs_grad});
        return use_attn_dropout ? output->dropout(dropout_rate_, training) : output;
    }
    const float keep_scale = 1.0f / (1.0f - dropout_rate_);

    Tensor causal_mask = Tensor::create_causal_mask(S);
    const float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_size));
    const float* mask_data = causal_mask.raw();

    // Q/K/V/concat live on the heap (shared_ptr) because the backward
    // closure needs them after forward returns.
    auto Q = std::make_shared<Tensor>(Tensor::uninitialized(flat, d));
    auto K = std::make_shared<Tensor>(Tensor::uninitialized(flat, d));
    auto V = std::make_shared<Tensor>(Tensor::uninitialized(flat, d));
    auto concat = std::make_shared<Tensor>(Tensor::uninitialized(flat, d));

    const float* in_data = input_tensor.raw();
    auto project = [&](const Tensor& W, const Tensor& b, Tensor& out) {
        fill_bias_rows(out.raw(), b.raw(), flat, d);
        blas_sgemm_ex(in_data, W.raw(), out.raw(), flat, d, d, false, false, 1.0f, 1.0f);
    };
    project(W_q->getData(), b_q->getData(), *Q);
    project(W_k->getData(), b_k->getData(), *K);
    project(W_v->getData(), b_v->getData(), *V);

    // RoPE: rotate Q and K in place, so both the score matmuls below
    // and the cached copies the backward pass regathers are the
    // rotated values. V is untouched.
    auto rope_cos = std::make_shared<std::vector<float>>();
    auto rope_sin = std::make_shared<std::vector<float>>();
    if (rope_) {
        build_rope_tables(S, head_size, *rope_cos, *rope_sin);
        for (size_t b = 0; b < batch_size; b++) {
            float* q_block = Q->raw() + b * S * d;
            float* k_block = K->raw() + b * S * d;
            rope_apply(q_block, S, d, H, head_size, rope_cos->data(), rope_sin->data(), false);
            rope_apply(k_block, S, d, H, head_size, rope_cos->data(), rope_sin->data(), false);
        }
    }

    // Softmax outputs (and dropout masks) per (batch, head), kept for the
    // backward pass only when there is one.
    const size_t cache_floats = batch_size * H * S * S;
    std::shared_ptr<std::vector<float>> attn_cache;
    std::shared_ptr<std::vector<float>> drop_mask;
    if (needs_grad) {
        attn_cache = std::make_shared<std::vector<float>>(cache_floats);
        if (use_attn_dropout) {
            drop_mask = std::make_shared<std::vector<float>>(cache_floats);
        }
    }

    const float* Q_data = Q->raw();
    const float* K_data = K->raw();
    const float* V_data = V->raw();
    float* concat_data = concat->raw();
    float* attn_data_all = attn_cache ? attn_cache->data() : nullptr;
    float* mask_all = drop_mask ? drop_mask->data() : nullptr;
    // One dropout stream per (batch, head) unit, reserved before the
    // parallel loop so each unit's mask is fixed by its index rather
    // than by which thread reached the RNG first.
    const uint64_t stream_base =
        use_attn_dropout ? reserve_dropout_streams(static_cast<uint64_t>(batch_size) * H) : 0;

    // Heads are column slices of the (rows, d_model) projections; BLAS
    // takes them as strided views (lda = d_model), so no per-head
    // gather/scatter copies are needed anywhere in this loop -
    // measured at ~12% of step CPU as memmove (BENCHMARKS.md #10).
    parallel_for(batch_size * H, 1, [&](size_t begin, size_t end) {
        Tensor scores = Tensor::uninitialized(S, S);
        // Per-task homes for what the cache would otherwise hold: the
        // softmax output when nothing is cached, the dropout mask when it
        // is not cached, and the dropped weights either way.
        Tensor attn_scratch = attn_data_all ? Tensor() : Tensor::uninitialized(S, S);
        Tensor mask_scratch =
            (use_attn_dropout && !mask_all) ? Tensor::uninitialized(S, S) : Tensor();
        Tensor dropped = use_attn_dropout ? Tensor::uninitialized(S, S) : Tensor();

        for (size_t unit = begin; unit < end; unit++) {
            const size_t b = unit / H;
            const size_t h = unit % H;
            const size_t batch_offset = b * S * d;
            const size_t col = h * head_size;

            const float* q_ptr = Q_data + batch_offset + col;
            const float* k_ptr = K_data + batch_offset + col;
            const float* v_ptr = V_data + batch_offset + col;

            float* scores_data = scores.raw();
            blas_sgemm_strided(false, true, S, S, head_size, 1.0f, q_ptr, d, k_ptr, d, 0.0f,
                               scores_data, S);
            for (size_t i = 0; i < S * S; i++) {
                scores_data[i] = scores_data[i] * scale_factor + mask_data[i];
            }

            // Softmax rows straight into the cache slice or scratch (same
            // math as Tensor::softmax: subtract max, vec_exp, scale by 1/sum).
            float* attn_unit = attn_data_all ? attn_data_all + unit * S * S : attn_scratch.raw();
            for (size_t i = 0; i < S; i++) {
                const float* row_in = scores_data + i * S;
                float* row_out = attn_unit + i * S;
                float max_val = row_in[0];
                for (size_t j = 1; j < S; j++) {
                    max_val = std::max(max_val, row_in[j]);
                }
                for (size_t j = 0; j < S; j++) {
                    row_out[j] = row_in[j] - max_val;
                }
                vec_exp(row_out, row_out, S);
                const float inv_sum = 1.0f / vec_sum(row_out, S);
                for (size_t j = 0; j < S; j++) {
                    row_out[j] *= inv_sum;
                }
            }

            const float* effective = attn_unit;
            if (use_attn_dropout) {
                float* m = mask_all ? mask_all + unit * S * S : mask_scratch.raw();
                fill_dropout_mask(m, S * S, dropout_rate_, keep_scale, stream_base + unit);
                float* w = dropped.raw();
                for (size_t i = 0; i < S * S; i++) {
                    w[i] = attn_unit[i] * m[i];
                }
                effective = w;
            }

            // attended lands directly in its concat column slice.
            blas_sgemm_strided(false, false, S, head_size, S, 1.0f, effective, S, v_ptr, d, 0.0f,
                               concat_data + batch_offset + col, d);
        }
    });

    // Output projection, also flat.
    Tensor out_tensor = Tensor::uninitialized(batch_size, S, d);
    fill_bias_rows(out_tensor.raw(), b_o->getData().raw(), flat, d);
    blas_sgemm_ex(concat_data, W_o->getData().raw(), out_tensor.raw(), flat, d, d, false, false,
                  1.0f, 1.0f);

    auto output = Variable::create(std::move(out_tensor), needs_grad);

    if (needs_grad) {
        auto self_Wq = W_q;
        auto self_Wk = W_k;
        auto self_Wv = W_v;
        auto self_Wo = W_o;
        auto self_bq = b_q;
        auto self_bk = b_k;
        auto self_bv = b_v;
        auto self_bo = b_o;

        const bool dropout_active = use_attn_dropout;
        const bool rope_active = rope_;
        output->setBackward(
            {input, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o},
            [self_input = input, self_Wq, self_Wk, self_Wv, self_Wo, self_bq, self_bk, self_bv,
             self_bo, Q, K, V, concat, attn_cache, drop_mask, rope_cos, rope_sin, rope_active,
             batch_size, S, d, H, head_size, flat, scale_factor, dropout_active](Variable& node) {
            // Each gradient is written only if its target requires grad:
            // ensureGrad leaves a frozen tensor's grad unallocated.
            const float* dOut = node.getGrad().raw();

            // Output projection gradients: single flat sgemms.
            // dWo += concat^T @ dOut ; dbo += column sums of dOut.
            if (self_Wo->requiresGrad()) {
                self_Wo->ensureGrad();
                blas_sgemm_ex(concat->raw(), dOut, self_Wo->getGrad().raw(), d, d, flat, true,
                              false, 1.0f, 1.0f);
            }
            if (self_bo->requiresGrad()) {
                self_bo->ensureGrad();
                add_column_sums(dOut, flat, d, self_bo->getGrad().raw());
            }

            // Everything below serves the Q/K/V projection weights and the
            // input; skip it when none of them wants a gradient.
            const bool grad_input = self_input->requiresGrad();
            if (!grad_input && !self_Wq->requiresGrad() && !self_Wk->requiresGrad()
                && !self_Wv->requiresGrad() && !self_bq->requiresGrad() && !self_bk->requiresGrad()
                && !self_bv->requiresGrad()) {
                return;
            }

            // dConcat = dOut @ Wo^T
            Tensor dConcat = Tensor::uninitialized(flat, d);
            blas_sgemm_ex(dOut, self_Wo->getData().raw(), dConcat.raw(), flat, d, d, false, true,
                          1.0f, 0.0f);

            Tensor dQ = Tensor::uninitialized(flat, d);
            Tensor dK = Tensor::uninitialized(flat, d);
            Tensor dV = Tensor::uninitialized(flat, d);
            float* dQ_data = dQ.raw();
            float* dK_data = dK.raw();
            float* dV_data = dV.raw();
            const float* dConcat_data = dConcat.raw();
            const float* Q_saved = Q->raw();
            const float* K_saved = K->raw();
            const float* V_saved = V->raw();
            const float* attn_all = attn_cache->data();
            const float* masks = drop_mask ? drop_mask->data() : nullptr;

            // Strided views throughout, mirroring forward: head slices
            // read with lda = d, and dQ/dK/dV written directly into
            // their disjoint column slices (each unit owns one).
            parallel_for(batch_size * H, 1, [&](size_t begin, size_t end) {
                Tensor dWeights = Tensor::uninitialized(S, S);
                Tensor dScores = Tensor::uninitialized(S, S);
                Tensor effective = Tensor::uninitialized(S, S);

                for (size_t unit = begin; unit < end; unit++) {
                    const size_t b = unit / H;
                    const size_t h = unit % H;
                    const size_t batch_offset = b * S * d;
                    const size_t col = h * head_size;

                    const float* q_ptr = Q_saved + batch_offset + col;
                    const float* k_ptr = K_saved + batch_offset + col;
                    const float* v_ptr = V_saved + batch_offset + col;
                    const float* dAtt_ptr = dConcat_data + batch_offset + col;

                    const float* W = attn_all + unit * S * S;  // pre-dropout softmax
                    const float* W_eff = W;
                    if (dropout_active) {
                        const float* m = masks + unit * S * S;
                        float* e = effective.raw();
                        for (size_t i = 0; i < S * S; i++) {
                            e[i] = W[i] * m[i];
                        }
                        W_eff = e;
                    }

                    // attended = W_eff @ V_head:
                    //   dW_eff = dAttended @ V_head^T ; dV_head = W_eff^T @ dAttended
                    blas_sgemm_strided(false, true, S, S, head_size, 1.0f, dAtt_ptr, d, v_ptr, d,
                                       0.0f, dWeights.raw(), S);
                    blas_sgemm_strided(true, false, S, head_size, S, 1.0f, W_eff, S, dAtt_ptr, d,
                                       0.0f, dV_data + batch_offset + col, d);

                    float* dW = dWeights.raw();
                    if (dropout_active) {
                        // Through the dropout: dW = dW_eff * mask.
                        const float* m = masks + unit * S * S;
                        for (size_t i = 0; i < S * S; i++) {
                            dW[i] *= m[i];
                        }
                    }

                    // Softmax backward per row:
                    //   dScores = W * (dW - sum(dW * W)) * scale
                    float* dS = dScores.raw();
                    for (size_t i = 0; i < S; i++) {
                        const float* w_row = W + i * S;
                        const float* dw_row = dW + i * S;
                        float* ds_row = dS + i * S;
                        float sum = 0.0f;
                        for (size_t j = 0; j < S; j++) {
                            sum += dw_row[j] * w_row[j];
                        }
                        for (size_t j = 0; j < S; j++) {
                            ds_row[j] = w_row[j] * (dw_row[j] - sum) * scale_factor;
                        }
                    }

                    // dQ_head = dScores @ K_head ; dK_head = dScores^T @ Q_head
                    blas_sgemm_strided(false, false, S, head_size, S, 1.0f, dS, S, k_ptr, d, 0.0f,
                                       dQ_data + batch_offset + col, d);
                    blas_sgemm_strided(true, false, S, head_size, S, 1.0f, dS, S, q_ptr, d, 0.0f,
                                       dK_data + batch_offset + col, d);
                }
            });

            // dQ/dK so far are gradients w.r.t. the ROTATED Q and K
            // (forward cached the rotated values). The inverse rotation
            // maps them back to pre-RoPE projection space before the
            // weight/bias/input gradients below.
            if (rope_active) {
                for (size_t b = 0; b < batch_size; b++) {
                    rope_apply(dQ_data + b * S * d, S, d, H, head_size, rope_cos->data(),
                               rope_sin->data(), true);
                    rope_apply(dK_data + b * S * d, S, d, H, head_size, rope_cos->data(),
                               rope_sin->data(), true);
                }
            }

            // Projection gradients, all flat single sgemms with beta=1
            // accumulation. dInput sums the three projection paths.
            const float* input_data = self_input->getData().raw();
            auto weight_grad = [&](const std::shared_ptr<Variable>& W, const float* dProj) {
                if (!W->requiresGrad()) return;
                W->ensureGrad();
                blas_sgemm_ex(input_data, dProj, W->getGrad().raw(), d, d, flat, true, false, 1.0f,
                              1.0f);
            };
            auto bias_grad = [&](const std::shared_ptr<Variable>& b, const float* dProj) {
                if (!b->requiresGrad()) return;
                b->ensureGrad();
                add_column_sums(dProj, flat, d, b->getGrad().raw());
            };
            weight_grad(self_Wq, dQ_data);
            weight_grad(self_Wk, dK_data);
            weight_grad(self_Wv, dV_data);
            bias_grad(self_bq, dQ_data);
            bias_grad(self_bk, dK_data);
            bias_grad(self_bv, dV_data);

            if (grad_input) {
                self_input->ensureGrad();
                float* dIn = self_input->getGrad().raw();
                blas_sgemm_ex(dQ_data, self_Wq->getData().raw(), dIn, flat, d, d, false, true, 1.0f,
                              1.0f);
                blas_sgemm_ex(dK_data, self_Wk->getData().raw(), dIn, flat, d, d, false, true, 1.0f,
                              1.0f);
                blas_sgemm_ex(dV_data, self_Wv->getData().raw(), dIn, flat, d, d, false, true, 1.0f,
                              1.0f);
            }
        });
    }

    auto final_output = output;
    if (training && dropout_rate_ > 0.0f) {
        final_output = output->dropout(dropout_rate_, training);
    }
    return final_output;
}

}  // namespace grad
