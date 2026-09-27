// Every resident-stream kernel (metal_ops.h) against the CPU: the CPU
// implementation itself where it is callable on its own (Tensor ops,
// fill_dropout_mask, LayerNorm and Variable ops in CPU mode), otherwise a
// loop with the CPU code's formula in the CPU code's order. Exits 77
// (skipped) without a Metal device.
//
// Tolerances. Elementwise kernels that round once per output (add, mul,
// scale, bias add, gathers, masks) must match exactly. Where an output is
// a multiply-add, either compiler may fuse it into one rounding (clang's
// default -ffp-contract=on, and Metal's), so the two sides can differ by an
// ulp of the product: an absolute bound of 2^-22 for products of magnitude
// below 2, since a relative bound fails wherever the sum cancels. Transcendentals (tanh, exp, log
// in safe math) are a few ulp from Accelerate's vForce: 1e-5 relative. Reductions reorder a sum of
// n terms, whose rounding error is at most ~n * 2^-24 of the sum of magnitudes; the bounds below
// are that, with margin, for the row lengths used (<= 1000).
#include "grad/transformer/activations.h"
#include "grad/transformer/device.h"
#include "grad/transformer/layer_norm.h"
#include "grad/transformer/metal_backend.h"
#include "grad/transformer/metal_ops.h"
#include "grad/transformer/tensor.h"
#include "grad/transformer/variable.h"
#include "../test_util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace grad;
namespace ops = grad::metal::ops;

namespace {

std::mt19937 gen(2468);

// An ulp of a float of magnitude below 2 (see the tolerance note above).
constexpr float kUlpOfProduct = 2.4e-7f;

// A Metal-mode tensor with U(lo, hi) values, written by the CPU (fresh, so
// no fence).
Tensor rnd(const Shape& shape, float lo = -1.0f, float hi = 1.0f) {
    Tensor t = Tensor::uninitialized(shape);
    std::uniform_real_distribution<float> dis(lo, hi);
    for (float& v : t.values()) v = dis(gen);
    return t;
}

std::vector<float> host(const Tensor& t) {
    return {t.values().begin(), t.values().end()};
}

// Runs fn with the process in CPU mode, for references computed by the CPU
// implementation itself.
void on_cpu(const std::function<void()>& fn) {
    set_device(Device::CPU);
    fn();
    set_device(Device::Metal);
}

// Worst |expected - actual| / (atol + rtol * |expected|); < 1 passes.
bool close(const std::string& what, const std::vector<float>& expected, const Tensor& actual_t,
           float atol, float rtol) {
    const std::vector<float> actual = host(actual_t);
    if (actual.size() != expected.size()) {
        std::cerr << what << ": size " << actual.size() << " vs " << expected.size() << std::endl;
        return CHECK(false);
    }
    float worst = 0.0f;
    for (size_t i = 0; i < actual.size(); i++) {
        const float tol = atol + rtol * std::abs(expected[i]);
        const float err = std::abs(expected[i] - actual[i]);
        worst = std::max(worst, tol > 0.0f ? err / tol : (err > 0.0f ? 2.0f : 0.0f));
        if (std::isnan(actual[i]) != std::isnan(expected[i])) worst = 2.0f;
    }
    std::cout << "  " << what << ": worst " << worst << " of tolerance" << std::endl;
    return CHECK(worst < 1.0f);
}

bool exact(const std::string& what, const std::vector<float>& expected, const Tensor& actual) {
    const std::vector<float> a = host(actual);
    const bool same = a.size() == expected.size()
                      && std::memcmp(a.data(), expected.data(), a.size() * sizeof(float)) == 0;
    std::cout << "  " << what << ": " << (same ? "bitwise equal" : "DIFFERENT") << std::endl;
    return CHECK(same);
}

void test_elementwise() {
    std::cout << "elementwise" << std::endl;
    constexpr size_t n = 5003;
    const Tensor a = rnd(Shape{n});
    const Tensor b = rnd(Shape{n});
    const std::vector<float> ha = host(a);
    const std::vector<float> hb = host(b);
    std::vector<float> ref(n);

    Tensor y = Tensor::uninitialized(Shape{n});
    ops::add(a.device_data(), b.device_data(), y.device_data(), n);
    for (size_t i = 0; i < n; i++) ref[i] = ha[i] + hb[i];
    exact("add", ref, y);

    ops::mul(a.device_data(), b.device_data(), y.device_data(), n);
    for (size_t i = 0; i < n; i++) ref[i] = ha[i] * hb[i];
    exact("mul", ref, y);

    ops::scale(a.device_data(), 0.37f, y.device_data(), n);
    for (size_t i = 0; i < n; i++) ref[i] = ha[i] * 0.37f;
    exact("scale", ref, y);

    ops::copy(a.device_data(), y.device_data(), n);
    exact("copy", ha, y);
    ops::accumulate(y.device_data(), b.device_data(), n);
    for (size_t i = 0; i < n; i++) ref[i] = ha[i] + hb[i];
    exact("accumulate", ref, y);

    ops::copy(a.device_data(), y.device_data(), n);
    ops::accumulate_scaled(y.device_data(), b.device_data(), -1.7f, n);
    for (size_t i = 0; i < n; i++) ref[i] = ha[i] + hb[i] * -1.7f;
    close("accumulate_scaled", ref, y, kUlpOfProduct, 1e-6f);

    ops::copy(a.device_data(), y.device_data(), n);
    ops::mul_accumulate(y.device_data(), a.device_data(), b.device_data(), n);
    for (size_t i = 0; i < n; i++) ref[i] = ha[i] + ha[i] * hb[i];
    close("mul_accumulate", ref, y, kUlpOfProduct, 1e-6f);

    Tensor s = Tensor::uninitialized(Shape{2});
    ops::fill(s.device_data(), 2, 0.25f);
    ops::copy(a.device_data(), y.device_data(), n);
    ops::scale_by(y.device_data(), s.device_data() + 1, n);
    for (size_t i = 0; i < n; i++) ref[i] = ha[i] * 0.25f;
    exact("scale_by", ref, y);
}

// Bias rows and the general broadcast, against Tensor::add on the CPU.
void test_broadcast_add() {
    std::cout << "broadcast add" << std::endl;
    const Tensor x = rnd(Shape{3, 7, 11});
    const Tensor bias = rnd(Shape{1, 11});
    const Tensor table = rnd(Shape{7, 11});
    const Tensor column = rnd(Shape{7, 1});
    std::vector<float> ref_bias, ref_table, ref_column;
    on_cpu([&] {
        ref_bias = host(x.add(bias));
        ref_table = host(x.add(table));
        ref_column = host(x.add(column));
    });
    Tensor y = Tensor::uninitialized(x.shape());
    ops::add_rows(x.device_data(), bias.device_data(), y.device_data(), x.numel(), 11, 1);
    exact("add_rows, bias", ref_bias, y);
    ops::add_rows(x.device_data(), table.device_data(), y.device_data(), x.numel(), 11, 7);
    exact("add_rows, position table", ref_table, y);
    ops::broadcast_add(x.device_data(), {77, 11, 1}, column.device_data(), {0, 1, 0},
                       y.device_data(), 3, 7, 11);
    exact("broadcast_add, column", ref_column, y);
}

void test_reductions() {
    std::cout << "column sums and broadcast reductions" << std::endl;
    for (const size_t rows : {size_t{5}, size_t{64}, size_t{2049}}) {
        constexpr size_t cols = 77;
        const Tensor g = rnd(Shape{rows, cols});
        Tensor dst = rnd(Shape{1, cols});
        const std::vector<float> hg = host(g);
        std::vector<float> ref = host(dst);
        std::vector<float> mag(cols, 0.0f);
        for (size_t r = 0; r < rows; r++) {
            for (size_t c = 0; c < cols; c++) {
                ref[c] += hg[r * cols + c];
                mag[c] += std::abs(hg[r * cols + c]);
            }
        }
        ops::column_sums(g.device_data(), rows, cols, dst.device_data());
        const float bound = static_cast<float>(rows) * 1.2e-7f * 2.0f;
        close("column_sums rows=" + std::to_string(rows), ref, dst,
              bound * *std::max_element(mag.begin(), mag.end()), 0.0f);
    }
    // (3, 5, 4) reduced to a (5, 1) column operand: batch and columns.
    const Tensor g = rnd(Shape{3, 5, 4});
    Tensor dst = rnd(Shape{5, 1});
    const std::vector<float> hg = host(g);
    std::vector<float> ref = host(dst);
    for (size_t i = 0; i < 5; i++) {
        float s = 0.0f;
        for (size_t b = 0; b < 3; b++) {
            for (size_t j = 0; j < 4; j++) s += hg[(b * 5 + i) * 4 + j];
        }
        ref[i] += s;
    }
    ops::broadcast_reduce(g.device_data(), 3, 5, 4, dst.device_data(), 5, 1, false, true);
    close("broadcast_reduce", ref, dst, 1e-6f, 1e-6f);
}

// GELU and SiLU against the CPU Variable ops, backward against the CPU
// formulas (activations use vForce on the CPU, the GPU's precise tanh/exp).
void test_activations() {
    std::cout << "activations" << std::endl;
    constexpr size_t n = 4099;
    const Tensor x = rnd(Shape{n}, -6.0f, 6.0f);
    const Tensor dy = rnd(Shape{n});
    const std::vector<float> hx = host(x);
    const std::vector<float> hdy = host(dy);
    std::vector<float> ref_gelu, ref_silu;
    on_cpu([&] {
        ref_gelu = host(Variable::create(x)->gelu()->getData());
        ref_silu = host(Variable::create(x)->silu()->getData());
    });
    Tensor y = Tensor::uninitialized(Shape{n});
    ops::gelu(x.device_data(), y.device_data(), n);
    close("gelu", ref_gelu, y, 1e-6f, 1e-5f);
    ops::silu(x.device_data(), y.device_data(), n);
    close("silu", ref_silu, y, 1e-6f, 1e-5f);

    constexpr float k = 0.79788456f;
    constexpr float a = 0.044715f;
    Tensor dx = rnd(Shape{n});
    std::vector<float> ref = host(dx);
    for (size_t i = 0; i < n; i++) {
        const float xi = hx[i];
        const float t = std::tanh(k * (xi + a * xi * xi * xi));
        const float d = 0.5f * (1.0f + t + xi * (1.0f - t * t) * k * (1.0f + 3.0f * a * xi * xi));
        ref[i] += d * hdy[i];
    }
    ops::gelu_backward(x.device_data(), dy.device_data(), dx.device_data(), n);
    close("gelu_backward", ref, dx, 1e-6f, 1e-5f);

    Tensor dx2 = rnd(Shape{n});
    ref = host(dx2);
    for (size_t i = 0; i < n; i++) {
        const float s = 1.0f / (1.0f + std::exp(-hx[i]));
        ref[i] += hdy[i] * s * (1.0f + hx[i] * (1.0f - s));
    }
    ops::silu_backward(x.device_data(), dy.device_data(), dx2.device_data(), n);
    close("silu_backward", ref, dx2, 1e-6f, 1e-5f);
}

// Forward against the CPU LayerNorm module; backward against the CPU
// module's formulas. Row width 700 is not a multiple of the 256-thread
// tree, and 70 rows span three column-sum blocks.
void test_norms() {
    for (const bool rms : {false, true}) {
        std::cout << (rms ? "rmsnorm" : "layernorm") << std::endl;
        constexpr size_t rows = 70;
        constexpr size_t d = 700;
        constexpr float eps = 1e-5f;
        const Tensor x = rnd(Shape{rows, d}, -2.0f, 3.0f);
        const Tensor gamma = rnd(Shape{1, d}, 0.5f, 1.5f);
        const Tensor beta = rnd(Shape{1, d});
        const Tensor dy = rnd(Shape{rows, d});

        std::vector<float> ref_y;
        on_cpu([&] {
            LayerNorm ln(static_cast<int>(d), rms);
            ln.setParams(gamma, beta);
            ref_y = host(ln.forward(Variable::create(x))->getData());
        });
        Tensor y = Tensor::uninitialized(x.shape());
        Tensor mean = Tensor::uninitialized(Shape{rows});
        Tensor rstd = Tensor::uninitialized(Shape{rows});
        ops::layer_norm(x.device_data(), gamma.device_data(), beta.device_data(), y.device_data(),
                        mean.device_data(), rstd.device_data(), rows, d, eps, rms);
        close("forward", ref_y, y, 1e-5f, 1e-5f);

        // The CPU backward (layer_norm.cpp), in double for a reference.
        const std::vector<float> hx = host(x), hg = host(gamma), hdy = host(dy);
        std::vector<double> dx_ref(rows * d, 0.0), dg_ref(d, 0.0), db_ref(d, 0.0);
        for (size_t r = 0; r < rows; r++) {
            const float* xr = hx.data() + r * d;
            const float* dyr = hdy.data() + r * d;
            double mu = 0.0, var = 0.0;
            if (!rms) {
                for (size_t j = 0; j < d; j++) mu += xr[j];
                mu /= d;
            }
            for (size_t j = 0; j < d; j++) var += (xr[j] - mu) * (xr[j] - mu);
            const double inv = 1.0 / std::sqrt(var / d + eps);
            double s1 = 0.0, s2 = 0.0;
            for (size_t j = 0; j < d; j++) {
                const double xhat = (xr[j] - mu) * inv;
                const double dn = static_cast<double>(dyr[j]) * hg[j];
                s1 += dn;
                s2 += dn * xhat;
                dg_ref[j] += dyr[j] * xhat;
                db_ref[j] += dyr[j];
            }
            for (size_t j = 0; j < d; j++) {
                const double xhat = (xr[j] - mu) * inv;
                const double dn = static_cast<double>(dyr[j]) * hg[j];
                dx_ref[r * d + j] =
                    rms ? inv * (dn - xhat * s2 / d) : inv * (dn - s1 / d - xhat * s2 / d);
            }
        }
        Tensor dx(x.shape());
        Tensor dgamma(Shape{1, d});
        Tensor dbeta(Shape{1, d});
        ops::layer_norm_backward(x.device_data(), gamma.device_data(), dy.device_data(),
                                 mean.device_data(), rstd.device_data(), dx.device_data(),
                                 dgamma.device_data(), dbeta.device_data(), rows, d, rms);
        const auto to_float = [](const std::vector<double>& v) {
            return std::vector<float>(v.begin(), v.end());
        };
        close("dx", to_float(dx_ref), dx, 1e-5f, 1e-4f);
        close("dgamma", to_float(dg_ref), dgamma, 1e-4f, 1e-4f);
        if (!rms) close("dbeta", to_float(db_ref), dbeta, 1e-4f, 1e-4f);
    }
}

void test_softmax_family() {
    std::cout << "softmax family" << std::endl;
    constexpr size_t rows = 33;
    constexpr size_t cols = 1000;
    const Tensor x = rnd(Shape{rows, cols}, -8.0f, 8.0f);
    const Tensor dy = rnd(Shape{rows, cols});
    std::vector<float> ref_soft, ref_log;
    on_cpu([&] {
        ref_soft = host(x.softmax());
        ref_log = host(Variable::create(x)->log_softmax()->getData());
    });
    Tensor y = Tensor::uninitialized(x.shape());
    ops::softmax(x.device_data(), y.device_data(), rows, cols);
    close("softmax", ref_soft, y, 1e-7f, 1e-5f);
    Tensor ly = Tensor::uninitialized(x.shape());
    ops::log_softmax(x.device_data(), ly.device_data(), rows, cols);
    close("log_softmax", ref_log, ly, 1e-5f, 1e-5f);

    const std::vector<float> hdy = host(dy);
    Tensor dx(x.shape());
    std::vector<float> ref(rows * cols);
    for (size_t r = 0; r < rows; r++) {
        double dot = 0.0;
        for (size_t c = 0; c < cols; c++) dot += ref_soft[r * cols + c] * hdy[r * cols + c];
        for (size_t c = 0; c < cols; c++) {
            ref[r * cols + c] =
                static_cast<float>(ref_soft[r * cols + c] * (hdy[r * cols + c] - dot));
        }
    }
    ops::softmax_backward(y.device_data(), dy.device_data(), dx.device_data(), rows, cols);
    close("softmax_backward", ref, dx, 1e-6f, 1e-4f);

    Tensor ldx(x.shape());
    for (size_t r = 0; r < rows; r++) {
        double s = 0.0;
        for (size_t c = 0; c < cols; c++) s += hdy[r * cols + c];
        for (size_t c = 0; c < cols; c++) {
            ref[r * cols + c] =
                static_cast<float>(hdy[r * cols + c] - std::exp(ref_log[r * cols + c]) * s);
        }
    }
    ops::log_softmax_backward(ly.device_data(), dy.device_data(), ldx.device_data(), rows, cols);
    close("log_softmax_backward", ref, ldx, 1e-5f, 1e-4f);
}

// The CPU attention path: scores * scale + (-1e9 above the diagonal), row
// softmax; backward through a dropout mask.
void test_attention_softmax() {
    std::cout << "attention softmax" << std::endl;
    constexpr size_t units = 6;
    constexpr size_t S = 37;
    constexpr float scale = 0.125f;
    const Tensor scores = rnd(Shape{units, S, S}, -4.0f, 4.0f);
    const std::vector<float> hs = host(scores);
    std::vector<float> P(units * S * S);
    for (size_t row = 0; row < units * S; row++) {
        const size_t i = row % S;
        std::vector<float> v(S);
        float m = -1e30f;
        for (size_t j = 0; j < S; j++) {
            v[j] = hs[row * S + j] * scale + (j > i ? -1e9f : 0.0f);
            m = std::max(m, v[j]);
        }
        double sum = 0.0;
        for (size_t j = 0; j < S; j++) sum += std::exp(v[j] - m);
        for (size_t j = 0; j < S; j++)
            P[row * S + j] = static_cast<float>(std::exp(v[j] - m) / sum);
    }
    Tensor p = scores;  // GPU copy; softmax runs in place
    ops::attention_softmax(p.device_data(), units * S, S, scale);
    close("forward", P, p, 1e-7f, 1e-5f);

    const Tensor dp_in = rnd(Shape{units, S, S});
    Tensor mask = Tensor::uninitialized(Shape{units, S, S});
    fill_dropout_mask(mask.raw(), mask.numel(), 0.25f, 1.0f / 0.75f, 12345);
    const std::vector<float> hdp = host(dp_in), hm = host(mask);
    std::vector<float> ref(units * S * S);
    for (size_t row = 0; row < units * S; row++) {
        double dot = 0.0;
        for (size_t j = 0; j < S; j++) dot += hdp[row * S + j] * hm[row * S + j] * P[row * S + j];
        for (size_t j = 0; j < S; j++) {
            const double g = static_cast<double>(hdp[row * S + j]) * hm[row * S + j];
            ref[row * S + j] = static_cast<float>(P[row * S + j] * (g - dot) * scale);
        }
    }
    Tensor dp = dp_in;
    ops::attention_softmax_backward(p.device_data(), dp.device_data(), mask.device_data(),
                                    units * S, S, scale);
    close("backward", ref, dp, 1e-6f, 1e-4f);
}

// NLL against Variable::nll_loss; the fused cross-entropy against the CPU
// log_softmax -> nll_loss composition, loss and logits gradient.
void test_losses() {
    std::cout << "losses" << std::endl;
    constexpr size_t rows = 50;
    constexpr size_t vocab = 97;
    const Tensor logits = rnd(Shape{rows, vocab}, -5.0f, 5.0f);
    Tensor targets = Tensor::uninitialized(Shape{rows, 1});
    std::uniform_int_distribution<int> tok(0, static_cast<int>(vocab) - 1);
    for (float& t : targets.values()) t = static_cast<float>(tok(gen));
    targets.raw()[7] = -1.0f;  // ignored, as on the CPU
    float ref_loss = 0.0f;
    float ref_nll = 0.0f;
    std::vector<float> ref_grad;
    on_cpu([&] {
        auto lv = Variable::create(logits, true);
        auto loss = lv->log_softmax()->nll_loss(Variable::create(targets));
        loss->backward();
        ref_loss = loss->getData().raw()[0];
        ref_grad = host(lv->getGrad());
        ref_nll = Variable::create(logits)->nll_loss(Variable::create(targets))->getData().raw()[0];
    });

    Tensor out = Tensor::uninitialized(Shape{1, 1});
    ops::nll_loss(logits.device_data(), targets.device_data(), out.device_data(), rows, vocab);
    close("nll_loss", {ref_nll}, out, 1e-6f, 1e-6f);

    Tensor stats = Tensor::uninitialized(Shape{rows, 2});
    Tensor row_loss = Tensor::uninitialized(Shape{rows});
    ops::cross_entropy(logits.device_data(), targets.device_data(), stats.device_data(),
                       row_loss.device_data(), out.device_data(), rows, vocab);
    close("cross_entropy", {ref_loss}, out, 1e-6f, 1e-6f);

    Tensor g = Tensor::uninitialized(Shape{1, 1});
    ops::fill(g.device_data(), 1, 1.0f);
    Tensor dlogits(logits.shape());
    ops::cross_entropy_backward(logits.device_data(), stats.device_data(), targets.device_data(),
                                g.device_data(), dlogits.device_data(), rows, vocab);
    close("cross_entropy_backward", ref_grad, dlogits, 1e-7f, 1e-5f);

    Tensor dlogp(logits.shape());
    ops::nll_loss_backward(g.device_data(), targets.device_data(), dlogp.device_data(), rows,
                           vocab);
    std::vector<float> ref_dlogp(rows * vocab, 0.0f);
    const std::vector<float> ht = host(targets);
    for (size_t r = 0; r < rows; r++) {
        const int t = static_cast<int>(ht[r]);
        if (t >= 0) ref_dlogp[r * vocab + static_cast<size_t>(t)] = -1.0f / rows;
    }
    exact("nll_loss_backward", ref_dlogp, dlogp);
}

Tensor index_tensor(const std::vector<uint32_t>& v) {
    Tensor t = Tensor::uninitialized(Shape{std::max<size_t>(v.size(), 1)});
    std::memcpy(t.raw(), v.data(), v.size() * sizeof(uint32_t));
    return t;
}

// Gather against the CPU lookup; scatter against the CPU loop's order
// (every position of a token, in increasing order), which it reproduces.
void test_embedding() {
    std::cout << "embedding" << std::endl;
    constexpr size_t vocab = 31;
    constexpr size_t d = 24;
    constexpr size_t tokens = 200;
    constexpr float scale = 1.0f;
    const Tensor table = rnd(Shape{vocab, d});
    Tensor ids = Tensor::uninitialized(Shape{tokens, 1});
    std::uniform_int_distribution<int> tok(0, 9);  // heavy repetition
    for (float& t : ids.values()) t = static_cast<float>(tok(gen) * 3);
    const std::vector<float> hids = host(ids), htab = host(table);

    Tensor out = Tensor::uninitialized(Shape{tokens, d});
    ops::embedding(ids.device_data(), table.device_data(), out.device_data(), tokens, d, scale);
    std::vector<float> ref(tokens * d);
    for (size_t t = 0; t < tokens; t++) {
        for (size_t j = 0; j < d; j++) {
            ref[t * d + j] = htab[static_cast<size_t>(hids[t]) * d + j] * scale;
        }
    }
    exact("forward", ref, out);

    // Segments by stable counting sort, as the graph code builds them.
    std::vector<uint32_t> count(vocab + 1, 0);
    for (float id : hids) count[static_cast<size_t>(id) + 1]++;
    std::vector<uint32_t> uniq, starts{0};
    for (size_t v = 0; v < vocab; v++) {
        if (count[v + 1]) {
            uniq.push_back(static_cast<uint32_t>(v));
            starts.push_back(starts.back() + count[v + 1]);
        }
    }
    std::vector<uint32_t> first(vocab, 0), order(tokens);
    for (size_t u = 0; u < uniq.size(); u++) first[uniq[u]] = starts[u];
    for (size_t t = 0; t < tokens; t++)
        order[first[static_cast<size_t>(hids[t])]++] = static_cast<uint32_t>(t);
    const Tensor tu = index_tensor(uniq), ts = index_tensor(starts), to = index_tensor(order);

    const Tensor dout = rnd(Shape{tokens, d});
    Tensor dtable = rnd(Shape{vocab, d});
    std::vector<float> ref_dt = host(dtable);
    const std::vector<float> hdo = host(dout);
    for (size_t t = 0; t < tokens; t++) {
        for (size_t j = 0; j < d; j++) {
            ref_dt[static_cast<size_t>(hids[t]) * d + j] += hdo[t * d + j] * scale;
        }
    }
    ops::embedding_backward(tu.device_data(), ts.device_data(), to.device_data(), uniq.size(),
                            dout.device_data(), dtable.device_data(), d, scale);
    close("backward", ref_dt, dtable, 0.0f, 1e-6f);
}

// RoPE against the CPU rotation (multihead_attention.cpp), and inverse
// after forward returns the input.
void test_rope() {
    std::cout << "rope" << std::endl;
    constexpr size_t B = 2, S = 13, H = 3, hs = 8, d = H * hs, half = hs / 2;
    Tensor cos_t = Tensor::uninitialized(Shape{S, half});
    Tensor sin_t = Tensor::uninitialized(Shape{S, half});
    for (size_t j = 0; j < half; j++) {
        const float theta =
            std::pow(10000.0f, -2.0f * static_cast<float>(j) / static_cast<float>(hs));
        for (size_t i = 0; i < S; i++) {
            cos_t.raw()[i * half + j] = std::cos(static_cast<float>(i) * theta);
            sin_t.raw()[i * half + j] = std::sin(static_cast<float>(i) * theta);
        }
    }
    const Tensor x = rnd(Shape{B * S, d});
    std::vector<float> ref = host(x);
    const std::vector<float> hc = host(cos_t), hsn = host(sin_t);
    for (size_t r = 0; r < B * S; r++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t j = 0; j < half; j++) {
                float* head = ref.data() + r * d + h * hs;
                const float c = hc[(r % S) * half + j], s = hsn[(r % S) * half + j];
                const float a = head[2 * j], b = head[2 * j + 1];
                head[2 * j] = a * c - b * s;
                head[2 * j + 1] = a * s + b * c;
            }
        }
    }
    Tensor y = x;
    ops::rope(y.device_data(), B * S, S, d, hs, cos_t.device_data(), sin_t.device_data(), false);
    close("forward", ref, y, kUlpOfProduct, 1e-6f);
    ops::rope(y.device_data(), B * S, S, d, hs, cos_t.device_data(), sin_t.device_data(), true);
    close("inverse of forward", host(x), y, 1e-6f, 1e-5f);
}

// Masks bitwise equal to fill_dropout_mask: sizes below a block, not a
// multiple of 4, spanning blocks, and several units from consecutive
// streams (attention's per-(batch, head) masks).
void test_dropout() {
    std::cout << "dropout" << std::endl;
    constexpr uint64_t seed = 0x1234ABCDull;
    set_dropout_seed(seed);
    struct Case {
        size_t n, units;
        uint64_t stream;
        float rate;
    };
    const Case cases[] = {{1, 1, 0, 0.1f},
                          {4099, 1, 3, 0.1f},
                          {65536, 1, 9, 0.5f},
                          {200003, 1, 1, 0.1f},
                          {9216, 12, 40, 0.1f}};
    for (const Case& c : cases) {
        const float scale = 1.0f / (1.0f - c.rate);
        std::vector<float> ref(c.n * c.units);
        for (size_t u = 0; u < c.units; u++) {
            fill_dropout_mask(ref.data() + u * c.n, c.n, c.rate, scale, c.stream + u);
        }
        const Tensor x = rnd(Shape{c.n * c.units});
        Tensor mask = Tensor::uninitialized(x.shape());
        Tensor y = Tensor::uninitialized(x.shape());
        ops::dropout(x.device_data(), mask.device_data(), y.device_data(), c.n, c.units,
                     current_dropout_seed(), c.stream, c.rate, scale);
        exact("mask n=" + std::to_string(c.n) + " units=" + std::to_string(c.units), ref, mask);
        const std::vector<float> hx = host(x);
        for (size_t i = 0; i < ref.size(); i++) ref[i] *= hx[i];
        exact("  y = x * mask", ref, y);
    }
}

// AdamW against the CPU update (optimizer.cpp), step 3 with decay.
void test_adamw() {
    std::cout << "adamw" << std::endl;
    constexpr size_t n = 3001;
    Tensor w = rnd(Shape{n});
    const Tensor g = rnd(Shape{n});
    Tensor m = rnd(Shape{n}, -0.1f, 0.1f);
    Tensor v = rnd(Shape{n}, 0.0f, 0.01f);
    const float b1 = 0.9f, b2 = 0.95f, lr = 3e-3f, eps = 1e-8f, wd = 0.1f;
    const float bc1 = static_cast<float>(1.0f - std::pow(b1, 3));
    const float bc2 = static_cast<float>(1.0f - std::pow(b2, 3));
    std::vector<float> rw = host(w), rm = host(m), rv = host(v);
    const std::vector<float> hg = host(g);
    for (size_t i = 0; i < n; i++) {
        const float gi = hg[i];
        const float mi = rm[i] = b1 * rm[i] + (1.0f - b1) * gi;
        const float vi = rv[i] = b2 * rv[i] + (1.0f - b2) * (gi * gi);
        rw[i] -= lr * ((mi * (1.0f / bc1)) / (std::sqrt(vi * (1.0f / bc2)) + eps) + wd * rw[i]);
    }
    ops::adamw(w.device_data(), g.device_data(), m.device_data(), v.device_data(), n,
               {lr, b1, b2, 1.0f / bc1, 1.0f / bc2, eps, wd});
    close("weights", rw, w, kUlpOfProduct, 1e-6f);
    close("m", rm, m, kUlpOfProduct, 1e-6f);
    close("v", rv, v, kUlpOfProduct, 1e-6f);
}

void test_grad_norm() {
    std::cout << "grad norm" << std::endl;
    const Tensor a = rnd(Shape{10000});
    const Tensor b = rnd(Shape{3});
    const Tensor c = rnd(Shape{4096});
    double sum = 0.0;
    for (const Tensor* t : {&a, &b, &c}) {
        for (float v : t->values()) sum += static_cast<double>(v) * v;
    }
    const auto norm = static_cast<float>(std::sqrt(sum));
    const ops::Span spans[] = {
        {a.device_data(), a.numel()}, {b.device_data(), b.numel()}, {c.device_data(), c.numel()}};
    Tensor out = Tensor::uninitialized(Shape{2});
    ops::grad_norm(spans, 10.0f, out.device_data());
    close("norm and clip coefficient", {norm, 10.0f / (norm + 1e-6f)}, out, 0.0f, 1e-5f);
    ops::grad_norm(spans, 1e9f, out.device_data());
    close("no clip", {norm, 1.0f}, out, 0.0f, 1e-5f);
}

// Reductions give the same bits every time.
void test_repeatability() {
    std::cout << "repeatability" << std::endl;
    const Tensor x = rnd(Shape{300, 777});
    const Tensor gamma = rnd(Shape{1, 777});
    const auto run = [&] {
        Tensor y = Tensor::uninitialized(x.shape());
        Tensor mean = Tensor::uninitialized(Shape{300});
        Tensor rstd = Tensor::uninitialized(Shape{300});
        ops::layer_norm(x.device_data(), gamma.device_data(), gamma.device_data(), y.device_data(),
                        mean.device_data(), rstd.device_data(), 300, 777, 1e-5f, false);
        Tensor dg(Shape{1, 777});
        ops::layer_norm_backward(x.device_data(), gamma.device_data(), x.device_data(),
                                 mean.device_data(), rstd.device_data(), nullptr, dg.device_data(),
                                 nullptr, 300, 777, false);
        Tensor cs(Shape{1, 777});
        ops::column_sums(x.device_data(), 300, 777, cs.device_data());
        std::vector<float> all = host(y);
        for (const Tensor* t : {&dg, &cs})
            all.insert(all.end(), t->values().begin(), t->values().end());
        return all;
    };
    const auto first = run();
    const auto second = run();
    CHECK(std::memcmp(first.data(), second.data(), first.size() * sizeof(float)) == 0);
}

}  // namespace

int main() {
    std::cout << "=== METAL KERNELS VS CPU ===" << std::endl;
    if (!metal::resident_available()) {
        std::cout << "Resident Metal unavailable (" << metal::resident_status() << "), skipping."
                  << std::endl;
        return 77;
    }
    set_device(Device::Metal);
    try {
        test_elementwise();
        test_broadcast_add();
        test_reductions();
        test_activations();
        test_norms();
        test_softmax_family();
        test_attention_softmax();
        test_losses();
        test_embedding();
        test_rope();
        test_dropout();
        test_adamw();
        test_grad_norm();
        test_repeatability();
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << std::endl;
        CHECK(false);
    }
    set_device(Device::CPU);
    return test_util::exit_code();
}
