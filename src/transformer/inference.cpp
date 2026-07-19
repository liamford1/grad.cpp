#include "transformer/inference.h"
#include "transformer/blas_wrapper.h"
#include <cmath>
#include <cstring>
#include <stdexcept>

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
}

const float* InferenceSession::step(int token_id) {
    if (pos_ >= max_len_) {
        throw std::runtime_error("InferenceSession: context exceeds max_len");
    }
    if (token_id < 0 || token_id >= vocab_) {
        throw std::out_of_range("InferenceSession: token id out of vocab range");
    }

    const Tensor& E = model_.getTokenEmbedding().getEmbeddingTable()->getData();
    const Tensor& P = model_.getPosEncoding().getPositionEmbeddings()->getData();
    const float emb_scale = model_.getTokenEmbedding().getScale();

    const float* e_row = E.raw() + static_cast<size_t>(token_id) * d_model_;
    const float* p_row = P.raw() + static_cast<size_t>(pos_) * d_model_;
    for (int j = 0; j < d_model_; j++) {
        x_[j] = e_row[j] * emb_scale + p_row[j];
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
        matvec(h_.data(), ffn.getLayer1Weights()->getData(),
               ffn.getLayer1Bias()->getData(), ffn_hidden_.data());
        gelu_row(ffn_hidden_.data(), d_ff_);
        matvec_add(ffn_hidden_.data(), ffn.getLayer2Weights()->getData(),
                   ffn.getLayer2Bias()->getData(), x_.data());
    }

    layer_norm_row(x_.data(), model_.getFinalNorm(), d_model_, h_.data());

    // Weight-tied output projection: logits = h @ E^T.
    blas_sgemm_ex(h_.data(), E.raw(), logits_.data(),
                  1, vocab_, d_model_, false, true, 1.0f, 0.0f);

    pos_++;
    return logits_.data();
}
