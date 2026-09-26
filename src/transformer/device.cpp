#include "grad/transformer/device.h"
#include "grad/transformer/metal_backend.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace grad {

void set_device(Device d) {
    if (d == current_device()) return;
    if (d == Device::Metal) {
        if (!metal::resident_available()) {
            throw std::runtime_error("Metal device unavailable: " + metal::resident_status());
        }
    } else {
        // Nothing queued may outlive the mode that queued it: CPU-mode ops
        // do not fence on tensors they did not hand to the GPU themselves.
        metal::synchronize();
    }
    device_detail::metal_active.store(d == Device::Metal, std::memory_order_relaxed);
}

const char* device_name(Device d) noexcept {
    return d == Device::Metal ? "metal" : "cpu";
}

std::optional<Device> parse_device(std::string_view name) noexcept {
    if (name == "cpu") return Device::CPU;
    if (name == "metal") return Device::Metal;
    return std::nullopt;
}

Device device_from_env() {
    const char* value = std::getenv("GRAD_DEVICE");
    if (!value || value[0] == '\0') return Device::CPU;
    if (const auto d = parse_device(value)) return *d;
    throw std::invalid_argument(std::string("GRAD_DEVICE must be cpu or metal, got '") + value
                                + "'");
}

}  // namespace grad
