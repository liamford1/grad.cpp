#pragma once

#include <atomic>
#include <optional>
#include <string_view>

// Execution device for the whole process (docs/design/metal-resident.md).
//
// CPU, the default, is the engine as it always was, including the
// threshold GEMM offload in metal_backend.h. Metal makes the process
// device-resident: tensors allocated from then on live in shared
// MTLBuffers, and Variable ops, the fused module ops and the optimizer
// encode their work into one GPU command stream instead of running it on
// the calling thread. The CPU waits only when it reads (or writes) a tensor
// the GPU may still be using; Tensor's accessors do that automatically.
//
// Select the device before building tensors: a tensor allocated in CPU mode
// cannot be handed to the GPU stream (the ops throw std::logic_error), while
// Metal-mode tensors stay usable from CPU code in either mode.
namespace grad {

enum class Device { CPU, Metal };

namespace device_detail {
inline std::atomic<bool> metal_active{false};
}  // namespace device_detail

// True when ops should encode GPU work. One relaxed load: read per op, and
// only set_device writes it.
[[nodiscard]] inline bool metal_mode() noexcept {
    return device_detail::metal_active.load(std::memory_order_relaxed);
}

[[nodiscard]] inline Device current_device() noexcept {
    return metal_mode() ? Device::Metal : Device::CPU;
}

// Switches the process to d. Metal throws std::runtime_error, with the
// reason, when there is no usable device (no Metal, GRAD_METAL=0, a GPU
// older than Apple family 7, or a kernel compile failure). Leaving Metal
// first waits for all queued GPU work.
void set_device(Device d);

// "cpu" or "metal".
[[nodiscard]] const char* device_name(Device d) noexcept;
// The device a name spells ("cpu", "metal"), or nullopt.
[[nodiscard]] std::optional<Device> parse_device(std::string_view name) noexcept;

// The device GRAD_DEVICE names, else CPU. Throws std::invalid_argument for
// any other value, so a typo is not silently a CPU run.
[[nodiscard]] Device device_from_env();

}  // namespace grad
