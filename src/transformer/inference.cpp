#include "transformer/inference.h"
#include "transformer/blas_wrapper.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

// y = x @ W + b for a single row x. W is (in, out) row-major, b is (1, out).
void matvec(const float* x, const Tensor& W, const Tensor& b, float* y) {
    const int in = W.getRows();
    const int out = W.getCols();
    std::memcpy(y, b.raw(), out * sizeof(float));
    blas_sgemm_ex(x, W.raw(), y, 1, out, in, false, false, 1.0f, 1.0f);
}

// y += x @ W + b (residual-add variant).
void matvec_add(const float* x, const Tensor& W, const Tensor& b, float* y) {
    const int in = W.getRows();
    const int out = W.getCols();
    const float* bias = b.raw();
    for (int j = 0; j < out; j++) y[j] += bias[j];
    blas_sgemm_ex(x, W.raw(), y, 1, out, in, false, false, 1.0f, 1.0f);
}

void layer_norm_row(const float* x, const LayerNorm& norm, int d, float* out) {
    const float* g = norm.getGamma()->getData().raw();

    if (norm.isRMS()) {
        float ms = 0.0f;
        for (int j = 0; j < d; j++) {
            ms += x[j] * x[j];
        }
        const float r = 1.0f / std::sqrt(ms / d + norm.getEpsilon());
        for (int j = 0; j < d; j++) {
            out[j] = g[j] * x[j] * r;
        }
        return;
    }

    const float* beta = norm.getBeta()->getData().raw();
    float mean = vec_sum(x, d) / d;
    float variance = 0.0f;
    for (int j = 0; j < d; j++) {
        float diff = x[j] - mean;
        variance += diff * diff;
    }
    variance /= d;
    const float inv_std = 1.0f / std::sqrt(variance + norm.getEpsilon());

    for (int j = 0; j < d; j++) {
        out[j] = g[j] * (x[j] - mean) * inv_std + beta[j];
    }
}

void softmax_row(float* s, int n) {
    float max_val = s[0];
    for (int t = 1; t < n; t++) max_val = std::max(max_val, s[t]);
    for (int t = 0; t < n; t++) s[t] -= max_val;
    vec_exp(s, s, n);
    const float inv_sum = 1.0f / vec_sum(s, n);
    for (int t = 0; t < n; t++) s[t] *= inv_sum;
}

void gelu_row(float* x, int n) {
    constexpr float k = 0.79788456f;
    constexpr float a = 0.044715f;
    std::vector<float> t(n);
    for (int i = 0; i < n; i++) {
        t[i] = k * (x[i] + a * x[i] * x[i] * x[i]);
    }
    vec_tanh(t.data(), t.data(), n);
    for (int i = 0; i < n; i++) {
        x[i] = 0.5f * x[i] * (1.0f + t[i]);
    }
}

void silu_row(float* x, int n) {
    std::vector<float> e(n);
    for (int i = 0; i < n; i++) {
        e[i] = -x[i];
    }
    vec_exp(e.data(), e.data(), n);
    for (int i = 0; i < n; i++) {
        x[i] = x[i] / (1.0f + e[i]);
    }
}

// y = x @ W for a single row, no bias (modern FFN projections).
void matvec_nobias(const float* x, const Tensor& W, float* y) {
    blas_sgemm_ex(x, W.raw(), y, 1, W.getCols(), W.getRows(), false, false, 1.0f, 0.0f);
}

// Rotates one row's heads by the position angles in cos/sin (head_size/2
// entries each) - the incremental-decoding counterpart of the training
// path's rope_apply. Must pair dims exactly the same way: (2j, 2j+1).
void rope_row(float* row, int num_heads, int head_size,
              const float* cos_p, const float* sin_p) {
    const int half = head_size / 2;
    for (int h = 0; h < num_heads; h++) {
        float* head = row + h * head_size;
        for (int j = 0; j < half; j++) {
            const float c = cos_p[j];
            const float s = sin_p[j];
            const float a = head[2 * j];
            const float b = head[2 * j + 1];
            head[2 * j] = a * c - b * s;
            head[2 * j + 1] = a * s + b * c;
        }
    }
}

}  // namespace

InferenceSession::InferenceSession(const GPTModel& model)
    : model_(model),
      d_model_(model.getDModel()),
      num_layers_(model.getNumLayers()),
      num_heads_(model.getNumHeads()),
      head_size_(model.getDModel() / model.getNumHeads()),
      d_ff_(model.getBlock(0).getFFN().getLayer1Weights()->getData().getCols()),
      max_len_(model.getMaxLen()),
      vocab_(model.getVocabSize()) {
    k_cache_.resize(num_layers_);
    v_cache_.resize(num_layers_);
    for (int l = 0; l < num_layers_; l++) {
        k_cache_[l].resize(static_cast<size_t>(max_len_) * d_model_);
        v_cache_[l].resize(static_cast<size_t>(max_len_) * d_model_);
    }
    x_.resize(d_model_);
    h_.resize(d_model_);
    q_.resize(d_model_);
    attn_.resize(d_model_);
    scores_.resize(max_len_);
    ffn_hidden_.resize(d_ff_);
    logits_.resize(vocab_);
    if (model.getArch() == GPTArch::Modern) {
        ffn_gate_.resize(d_ff_);
        rope_cos_.resize(head_size_ / 2);
        rope_sin_.resize(head_size_ / 2);
    }
}

const float* InferenceSession::step(int token_id) {
    if (pos_ >= max_len_) {
        throw std::runtime_error("InferenceSession: context exceeds max_len");
    }
    if (token_id < 0 || token_id >= vocab_) {
        throw std::out_of_range("InferenceSession: token id out of vocab range");
    }

    const bool modern = model_.getArch() == GPTArch::Modern;
    const Tensor& E = model_.getTokenEmbedding().getEmbeddingTable()->getData();
    const float emb_scale = model_.getTokenEmbedding().getScale();

    const float* e_row = E.raw() + static_cast<size_t>(token_id) * d_model_;
    if (modern) {
        // Position comes from RoPE inside attention; precompute this
        // position's rotation angles once for all layers.
        for (int j = 0; j < d_model_; j++) {
            x_[j] = e_row[j] * emb_scale;
        }
        const int half = head_size_ / 2;
        for (int j = 0; j < half; j++) {
            const float theta = std::pow(10000.0f, -2.0f * j / head_size_);
            rope_cos_[j] = std::cos(pos_ * theta);
            rope_sin_[j] = std::sin(pos_ * theta);
        }
    } else {
        const Tensor& P = model_.getPosEncoding().getPositionEmbeddings()->getData();
        const float* p_row = P.raw() + static_cast<size_t>(pos_) * d_model_;
        for (int j = 0; j < d_model_; j++) {
            x_[j] = e_row[j] * emb_scale + p_row[j];
        }
    }

    const float inv_sqrt_hs = 1.0f / std::sqrt(static_cast<float>(head_size_));
    const int ctx = pos_ + 1;

    for (int l = 0; l < num_layers_; l++) {
        const TransformerBlock& blk = model_.getBlock(l);
        const MultiHeadAttention& att = blk.getAttention();

        // Pre-norm attention. K/V for this position go straight into the
        // cache; attention then runs against all cached positions.
        layer_norm_row(x_.data(), blk.getNorm1(), d_model_, h_.data());

        float* k_row = k_cache_[l].data() + static_cast<size_t>(pos_) * d_model_;
        float* v_row = v_cache_[l].data() + static_cast<size_t>(pos_) * d_model_;
        matvec(h_.data(), att.getW_q()->getData(), att.getB_q()->getData(), q_.data());
        matvec(h_.data(), att.getW_k()->getData(), att.getB_k()->getData(), k_row);
        matvec(h_.data(), att.getW_v()->getData(), att.getB_v()->getData(), v_row);

        // RoPE: rotate q and this position's k before caching, matching
        // the training path (the cache holds rotated keys).
        if (modern) {
            rope_row(q_.data(), num_heads_, head_size_, rope_cos_.data(), rope_sin_.data());
            rope_row(k_row, num_heads_, head_size_, rope_cos_.data(), rope_sin_.data());
        }

        const float* K = k_cache_[l].data();
        const float* V = v_cache_[l].data();

        for (int hd = 0; hd < num_heads_; hd++) {
            const int off = hd * head_size_;
            const float* qh = q_.data() + off;

            for (int t = 0; t < ctx; t++) {
                scores_[t] = cblas_sdot(head_size_, qh, 1,
                                        K + static_cast<size_t>(t) * d_model_ + off, 1)
                             * inv_sqrt_hs;
            }
            softmax_row(scores_.data(), ctx);

            float* out_h = attn_.data() + off;
            std::memset(out_h, 0, head_size_ * sizeof(float));
            for (int t = 0; t < ctx; t++) {
                cblas_saxpy(head_size_, scores_[t],
                            V + static_cast<size_t>(t) * d_model_ + off, 1, out_h, 1);
            }
        }

        // Output projection + residual, then pre-norm FFN + residual.
        matvec_add(attn_.data(), att.getW_o()->getData(), att.getB_o()->getData(), x_.data());

        layer_norm_row(x_.data(), blk.getNorm2(), d_model_, h_.data());
        const FeedForward& ffn = blk.getFFN();
        if (modern) {
            // SwiGLU: silu(gate(h)) * up(h) -> down, all bias-free.
            matvec_nobias(h_.data(), ffn.getGateWeights()->getData(), ffn_gate_.data());
            silu_row(ffn_gate_.data(), d_ff_);
            matvec_nobias(h_.data(), ffn.getLayer1Weights()->getData(), ffn_hidden_.data());
            for (int j = 0; j < d_ff_; j++) {
                ffn_hidden_[j] *= ffn_gate_[j];
            }
            blas_sgemm_ex(ffn_hidden_.data(), ffn.getLayer2Weights()->getData().raw(),
                          x_.data(), 1, d_model_, d_ff_, false, false, 1.0f, 1.0f);
        } else {
            matvec(h_.data(), ffn.getLayer1Weights()->getData(),
                   ffn.getLayer1Bias()->getData(), ffn_hidden_.data());
            gelu_row(ffn_hidden_.data(), d_ff_);
            matvec_add(ffn_hidden_.data(), ffn.getLayer2Weights()->getData(),
                       ffn.getLayer2Bias()->getData(), x_.data());
        }
    }

    layer_norm_row(x_.data(), model_.getFinalNorm(), d_model_, h_.data());

    // Weight-tied output projection: logits = h @ E^T.
    blas_sgemm_ex(h_.data(), E.raw(), logits_.data(),
                  1, vocab_, d_model_, false, true, 1.0f, 0.0f);

    pos_++;
    return logits_.data();
}
