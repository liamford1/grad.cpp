#pragma once

// Minimal failure-counting checks for the framework-free test executables.
// Each failed check prints file:line and the failing expression, and bumps
// a per-executable counter; main() returns test_util::exit_code() so ctest
// sees any failure. Checks do not abort, so one run reports every failure.

#include <cmath>
#include <iostream>

namespace test_util {

inline int& failure_count() {
    static int count = 0;
    return count;
}

inline bool record(bool ok, const char* expr, const char* file, int line) {
    if (!ok) {
        ++failure_count();
        std::cerr << file << ":" << line << ": CHECK failed: " << expr << std::endl;
    }
    return ok;
}

inline bool record_near(double actual, double expected, double tol, const char* expr,
                        const char* file, int line) {
    // Written so a NaN on either side fails rather than slipping through a
    // negated comparison.
    const bool ok = std::abs(actual - expected) <= tol;
    if (!ok) {
        ++failure_count();
        std::cerr << file << ":" << line << ": CHECK_NEAR failed: " << expr << " (actual " << actual
                  << ", expected " << expected << ", tolerance " << tol << ")" << std::endl;
    }
    return ok;
}

// Prints a one-line verdict and returns the process exit code.
inline int exit_code() {
    const int failures = failure_count();
    if (failures == 0) {
        std::cout << "\nAll checks passed" << std::endl;
        return 0;
    }
    std::cerr << "\n" << failures << " check(s) FAILED" << std::endl;
    return 1;
}

}  // namespace test_util

#define CHECK(cond) ::test_util::record(static_cast<bool>(cond), #cond, __FILE__, __LINE__)
#define CHECK_NEAR(actual, expected, tol)                                                  \
    ::test_util::record_near(static_cast<double>(actual), static_cast<double>(expected),   \
                             static_cast<double>(tol), #actual " ~= " #expected, __FILE__, \
                             __LINE__)
