// GRAD_METAL_CHECK (op_check.h): a non-finite forward output and a
// non-finite gradient are each reported against the op that made them, on
// the CPU and, when there is a Metal device, on the GPU.
#include "grad/transformer/device.h"
#include "grad/transformer/metal_backend.h"
#include "grad/transformer/op_check.h"
#include "grad/transformer/tensor.h"
#include "grad/transformer/variable.h"
#include "../test_util.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

using namespace grad;

namespace {

std::string g_dump_dir;

std::string manifest() {
    std::ifstream in(g_dump_dir + "/manifest.txt");
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

Tensor row(float a, float b) {
    Tensor t(1, 2);
    t.setValue(0, 0, a);
    t.setValue(0, 1, b);
    return t;
}

// Runs fn, which must throw the check's error, and returns the dump's
// manifest (its first line is the report).
std::string expect_report(const char* what, const std::function<void()>& fn) {
    std::filesystem::remove(g_dump_dir + "/manifest.txt");
    op_check::set_step(7);
    bool thrown = false;
    try {
        fn();
    } catch (const std::runtime_error& e) {
        thrown = std::string(e.what()).find("GRAD_METAL_CHECK") != std::string::npos;
    }
    std::cout << "  " << what << ": " << (thrown ? "reported" : "NOT reported") << std::endl;
    CHECK(thrown);
    return manifest();
}

void run(const char* device) {
    std::cout << device << std::endl;
    // 2e38 * 10 overflows in the forward pass.
    const std::string fwd = expect_report("forward overflow", [] {
        auto x = Variable::create(row(1.0f, 2e38f), true);
        auto y = x->scale(10.0f);
    });
    CHECK(fwd.find("step 7, forward op 1, ") != std::string::npos);
    CHECK(fwd.find("scale") != std::string::npos);
    CHECK(fwd.find("(0, 1) = inf") != std::string::npos);
    CHECK(fwd.find("input 0 (1, 2): finite") != std::string::npos);

    // Finite forward (z = 1, loss = 2e10), but dx = 1e30 * 1e10 overflows:
    // the scale's backward is at fault, not the matmul's.
    const std::string bwd = expect_report("backward overflow", [] {
        auto x = Variable::create(row(1e-30f, 1e-30f), true);
        Tensor wt(2, 1);
        wt.setValue(0, 0, 1e10f);
        wt.setValue(1, 0, 1e10f);
        auto w = Variable::create(wt, false);
        auto loss = x->scale(1e30f)->matmul(w);
        loss->backward();
    });
    CHECK(bwd.find("step 7, backward op 2, ") != std::string::npos);
    CHECK(bwd.find("scale") != std::string::npos);
    CHECK(bwd.find("output gradient (1, 2): finite") != std::string::npos);
    CHECK(std::filesystem::exists(g_dump_dir + "/out_grad.f32"));
    CHECK(std::filesystem::exists(g_dump_dir + "/in0_grad.f32"));

    // Finite values pass silently.
    bool clean = true;
    try {
        auto x = Variable::create(row(1.0f, 2.0f), true);
        auto loss = x->scale(3.0f)->matmul(Variable::create(Tensor(2, 1), false));
        loss->backward();
    } catch (const std::exception&) {
        clean = false;
    }
    CHECK(clean);
}

}  // namespace

int main() {
    std::cout << "=== GRAD_METAL_CHECK ===" << std::endl;
    char tmpl[] = "/tmp/grad_op_check_XXXXXX";
    if (!mkdtemp(tmpl)) return 1;
    g_dump_dir = tmpl;
    // Read once, at the first op.
    setenv("GRAD_METAL_CHECK", "1", 1);
    setenv("GRAD_METAL_CHECK_DUMP", g_dump_dir.c_str(), 1);
    CHECK(op_check::enabled());

    run("cpu");
    if (metal::resident_available()) {
        set_device(Device::Metal);
        run("metal");
        set_device(Device::CPU);
    } else {
        std::cout << "metal: unavailable, skipped" << std::endl;
    }
    std::filesystem::remove_all(g_dump_dir);
    return test_util::exit_code();
}
