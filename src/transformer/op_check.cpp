#include "grad/transformer/op_check.h"
#include "grad/transformer/device.h"
#include "grad/transformer/metal_backend.h"
#include "grad/transformer/variable.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cxxabi.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace grad::op_check {

namespace {

struct Config {
    int from_step = 0;
    std::string dump_dir;
};

Config& config() {
    static Config c = [] {
        Config out;
        if (const char* from = std::getenv("GRAD_METAL_CHECK_FROM"))
            out.from_step = std::atoi(from);
        if (const char* dir = std::getenv("GRAD_METAL_CHECK_DUMP")) out.dump_dir = dir;
        return out;
    }();
    return c;
}

// Only the training thread builds graphs, so plain state suffices.
int g_step = -1;
// Ops checked so far in this step, by phase: which of the step's many
// same-shaped ops (one per layer) a report is about.
size_t g_forward_ops = 0;
size_t g_backward_ops = 0;
std::vector<Tensor> g_grads_before;

bool active() {
    return g_step < 0 || g_step >= config().from_step;
}

// The function that built the op, from its backward closure's type:
// "std::shared_ptr<grad::Variable> grad::metal_graph::gelu(...)::'lambda'
// (grad::Variable&)" and the like, cut before the closure, without
// namespaces that say nothing, parameter lists or the return type.
std::string op_name(const std::type_info& op) {
    int status = 0;
    char* demangled = abi::__cxa_demangle(op.name(), nullptr, nullptr, &status);
    std::string name = (status == 0 && demangled) ? demangled : op.name();
    std::free(demangled);
    for (const char* marker : {"::'lambda", "::$_", "::{lambda"}) {
        const size_t at = name.find(marker);
        if (at != std::string::npos) {
            name.resize(at);
            break;
        }
    }
    for (const std::string prefix : {"(anonymous namespace)::", "grad::", "std::__1::"}) {
        for (size_t at = name.find(prefix); at != std::string::npos; at = name.find(prefix)) {
            name.erase(at, prefix.size());
        }
    }
    std::string out;
    int parens = 0;
    for (const char c : name) {
        if (c == '(') parens++;
        if (parens == 0) out += c;
        if (c == ')') parens--;
    }
    // The return type ends at the last space outside template arguments.
    int angles = 0;
    size_t begin = 0;
    for (size_t i = 0; i < out.size(); i++) {
        if (out[i] == '<') angles++;
        if (out[i] == '>') angles--;
        if (out[i] == ' ' && angles == 0) begin = i + 1;
    }
    return out.substr(begin);
}

struct Finding {
    size_t first = 0;
    size_t count = 0;
    float value = 0.0f;
};

// Waits for queued GPU work (rethrowing a command buffer failure) and
// scans t; count is 0 when every element is finite.
Finding scan(const Tensor& t) {
    Finding f;
    if (t.numel() == 0) return f;
    const float* p = t.raw();
    for (size_t i = 0; i < t.numel(); i++) {
        if (!std::isfinite(p[i])) {
            if (f.count == 0) {
                f.first = i;
                f.value = p[i];
            }
            f.count++;
        }
    }
    return f;
}

std::string coords(const Shape& shape, size_t flat) {
    std::string out = "(";
    size_t stride = shape.numel();
    for (size_t axis = 0; axis < shape.rank(); axis++) {
        stride /= shape[axis];
        out += std::to_string(flat / stride);
        flat %= stride;
        if (axis + 1 < shape.rank()) out += ", ";
    }
    return out + ")";
}

void sync() {
    if (metal_mode()) metal::synchronize();
}

std::string describe(std::string_view what, const Tensor& t, const Finding& f) {
    std::ostringstream os;
    os << what << " " << t.shape().to_string() << ": " << f.count << " non-finite of " << t.numel()
       << ", first at " << f.first << " " << coords(t.shape(), f.first) << " = " << f.value;
    return os.str();
}

// Each input's data (and gradient, before and after, for a backward) as
// DIR/<k>_<role>.f32, listed with shapes in DIR/manifest.txt.
void dump(const std::string& header, const Variable* out,
          std::span<const std::shared_ptr<Variable>> inputs, bool backward) {
    const std::string& dir = config().dump_dir;
    if (dir.empty()) return;
    std::ofstream manifest(dir + "/manifest.txt");
    manifest << header << "\n";
    const auto write = [&](const std::string& file, const Tensor& t) {
        if (t.numel() == 0) return;
        std::ofstream f(dir + "/" + file, std::ios::binary);
        f.write(reinterpret_cast<const char*>(t.raw()),
                static_cast<std::streamsize>(t.numel() * sizeof(float)));
        manifest << file << " " << t.shape().to_string() << "\n";
    };
    if (out) {
        write("out_data.f32", out->getData());
        if (backward) write("out_grad.f32", out->getGrad());
    }
    for (size_t k = 0; k < inputs.size(); k++) {
        const std::string base = "in" + std::to_string(k);
        write(base + "_data.f32", inputs[k]->getData());
        if (!backward) continue;
        if (k < g_grads_before.size()) write(base + "_grad_before.f32", g_grads_before[k]);
        write(base + "_grad.f32", inputs[k]->getGrad());
    }
    std::cerr << "GRAD_METAL_CHECK: tensors written to " << dir << std::endl;
}

// Reports whether each input's data was finite: if all were, the op made
// the bad value.
std::string input_summary(std::span<const std::shared_ptr<Variable>> inputs) {
    std::ostringstream os;
    for (size_t k = 0; k < inputs.size(); k++) {
        const Tensor& t = inputs[k]->getData();
        const Finding f = scan(t);
        os << "\n  input " << k << " " << t.shape().to_string() << ": "
           << (f.count == 0 ? "finite" : describe("data", t, f));
    }
    return os.str();
}

[[noreturn]] void fail(const std::string& report) {
    std::cerr << report << std::endl;
    throw std::runtime_error("GRAD_METAL_CHECK: non-finite value (see report above)");
}

std::string at_step() {
    return g_step < 0 ? "GRAD_METAL_CHECK: "
                      : "GRAD_METAL_CHECK: step " + std::to_string(g_step) + ", ";
}

std::string where(const char* phase, size_t index, const std::type_info& op) {
    return at_step() + phase + " op " + std::to_string(index) + ", " + op_name(op);
}

}  // namespace

namespace detail {
bool read_env() {
    const char* value = std::getenv("GRAD_METAL_CHECK");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}
}  // namespace detail

void set_step(int step) {
    g_step = step;
    g_forward_ops = 0;
    g_backward_ops = 0;
}

void forward(const Variable& out, std::span<const std::shared_ptr<Variable>> inputs,
             const std::type_info& op) {
    if (!active()) return;
    g_forward_ops++;
    sync();
    const Finding f = scan(out.getData());
    if (f.count == 0) return;
    const std::string head = where("forward", g_forward_ops, op);
    const std::string report =
        head + "\n  " + describe("output", out.getData(), f) + input_summary(inputs);
    dump(report, &out, inputs, false);
    fail(report);
}

void before_backward(std::span<const std::shared_ptr<Variable>> inputs) {
    g_grads_before.clear();
    if (!active() || config().dump_dir.empty()) return;
    for (const auto& in : inputs) g_grads_before.push_back(in->getGrad());
}

void backward(const Variable& out, std::span<const std::shared_ptr<Variable>> inputs,
              const std::type_info& op) {
    if (!active()) return;
    g_backward_ops++;
    sync();
    for (size_t k = 0; k < inputs.size(); k++) {
        const Tensor& g = inputs[k]->getGrad();
        const Finding f = scan(g);
        if (f.count == 0) continue;
        const Finding dy = scan(out.getGrad());
        std::ostringstream os;
        os << where("backward", g_backward_ops, op) << "\n  " << describe("gradient of input", g, f)
           << " (input " << k << ")\n  output gradient " << out.getGrad().shape().to_string()
           << ": " << (dy.count == 0 ? "finite" : describe("", out.getGrad(), dy))
           << input_summary(inputs);
        if (k < g_grads_before.size()) {
            const Finding before = scan(g_grads_before[k]);
            os << "\n  that gradient before this op: "
               << (before.count == 0 ? "finite" : describe("", g_grads_before[k], before));
        }
        dump(os.str(), &out, inputs, true);
        fail(os.str());
    }
    g_grads_before.clear();
}

void tensor(std::string_view what, const Tensor& t) {
    if (!active()) return;
    sync();
    const Finding f = scan(t);
    if (f.count == 0) return;
    fail(at_step() + "outside the graph\n  " + describe(what, t, f));
}

}  // namespace grad::op_check
