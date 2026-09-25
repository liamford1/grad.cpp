#pragma once

#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace grad {

// Integer conversion that throws std::overflow_error instead of silently
// changing the value (a negative int to size_t, a size_t past INT_MAX to
// int). For boundaries, not inner loops: converting a validated int
// hyperparameter to an extent once, or a size_t extent to the int a BLAS
// or file format takes.
template <typename To, typename From>
[[nodiscard]] constexpr To narrow(From value) {
    static_assert(std::is_integral_v<To> && std::is_integral_v<From>,
                  "grad::narrow converts between integer types");
    if (!std::in_range<To>(value)) {
        throw std::overflow_error("grad::narrow: " + std::to_string(value)
                                  + " does not fit the target integer type");
    }
    return static_cast<To>(value);
}

}  // namespace grad
