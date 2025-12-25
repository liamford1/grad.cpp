# Building Your CUDA Transformer

## Prerequisites

### 1. NVIDIA GPU & Driver
```bash
# Check if you have a CUDA-capable GPU
nvidia-smi
```

You should see your GPU info. If not, install NVIDIA drivers first.

### 2. CUDA Toolkit
Download from: https://developer.nvidia.com/cuda-downloads

```bash
# Verify installation
nvcc --version
```

### 3. CMake
You need CMake ≥ 3.18 for CUDA support.

```bash
cmake --version
```

---

## Build Instructions

### Clean Build (Recommended First Time)

```bash
# Remove old CPU-only build
rm -rf build/
mkdir build && cd build

# Configure with CUDA
cmake ..

# You should see:
#   -- CUDA Version: X.X
#   -- Using CUDA architectures: 75;86;89

# Build everything
cmake --build . -j$(nproc)
```

### GPU Architecture Tuning

In `CMakeLists.txt:16`, adjust for your GPU:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
```

**Common GPUs:**
- RTX 2060/2070/2080: `75` (Turing)
- RTX 3060/3070/3080/3090: `86` (Ampere)
- RTX 4060/4070/4080/4090: `89` (Ada Lovelace)
- A100: `80` (Ampere)
- H100: `90` (Hopper)

To build for **only your GPU** (faster compile):
```cmake
set(CMAKE_CUDA_ARCHITECTURES 86)  # Example for RTX 3080
```

---

## Testing Your CUDA Setup

### Step 1: Basic Infrastructure Test

```bash
cd build
./test_cuda_basic
```

**Expected output:**
```
=== CUDA Tensor Infrastructure Test ===

[Test 1] Creating CPU tensor...
  ✓ CPU tensor created: shape [2, 3]
  ✓ Device: CPU
  ✓ Elements: 6

[Test 2] Creating CUDA tensor...
  ✓ CUDA tensor created: shape [4, 5]
  ✓ Device: CUDA
  ✓ Elements: 20

...
=== ALL TESTS PASSED ===
```

If this passes, your CUDA infrastructure is working!

### Step 2: cuBLAS Matrix Multiplication Test

```bash
./test_matmul
```

**Expected output:**
```
=== cuBLAS Matrix Multiplication Test ===

[Test 1] 2x2 Matrix Multiplication
A [2x2]:
    1.000   2.000
    3.000   4.000
B [2x2]:
    5.000   6.000
    7.000   8.000
C = A @ B [2x2]:
   19.000  22.000
   43.000  50.000
  ✓ Test 1 PASSED

[Test 2] Non-Square Matrix Multiplication [3x2] @ [2x4]
  ✓ Test 2 PASSED

[Test 3] Batched Matrix Multiplication [2, 2, 2] @ [2, 2, 2]
  ✓ Test 3 PASSED (Batched multiplication works!)

[Test 4] Performance Benchmark - Large Matrix (1024x1024)
  Matrix size: 1024x1024
  Time: 2.3 ms
  Performance: 932 GFLOPS
  ✓ Test 4 PASSED
  ✓ Correctness verified

=== ALL MATMUL TESTS PASSED ===
```

**Performance benchmarks:**
- RTX 3080: ~15-30 TFLOPS theoretical, expect 5-10 TFLOPS in practice
- RTX 4090: ~82 TFLOPS theoretical (FP32)
- A100: ~19.5 TFLOPS (FP32), ~312 TFLOPS (TF32)

If you see >500 GFLOPS, your GPU is working properly!

---

## Common Build Errors

### Error: `nvcc: command not found`

**Fix:** Add CUDA to your PATH:

```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Error: `CUDA architectures not set`

**Fix:** CMake couldn't detect your GPU. Manually set in CMakeLists.txt:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 86)  # Use your GPU's compute capability
```

### Error: `cudaErrorNoKernelImageForDevice`

**Cause:** You compiled for the wrong GPU architecture.

**Fix:** Check your GPU's compute capability:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

Then update `CMAKE_CUDA_ARCHITECTURES` to match.

### Error: Linker cannot find `-lcudart`

**Fix:** Install CUDA Toolkit properly, then:

```bash
export CUDACXX=/usr/local/cuda/bin/nvcc
cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..
```

---

## Performance Verification

### CPU vs CUDA Fill Kernel

Create a simple benchmark:

```cpp
#include "transformer/tensor.h"
#include <chrono>

int main() {
    const int size = 10000000;  // 10M elements

    // CPU
    Tensor cpu_t({size}, Device::CPU);
    auto start = std::chrono::high_resolution_clock::now();
    cpu_t.fill(1.0f);
    auto end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration<double, std::milli>(end - start).count();

    // CUDA
    Tensor cuda_t({size}, Device::CUDA);
    start = std::chrono::high_resolution_clock::now();
    cuda_t.fill(1.0f);
    cudaDeviceSynchronize();  // Wait for GPU
    end = std::chrono::high_resolution_clock::now();
    auto cuda_time = std::chrono::duration<double, std::milli>(end - start).count();

    printf("CPU:  %.2f ms\n", cpu_time);
    printf("CUDA: %.2f ms (%.1fx faster)\n", cuda_time, cpu_time / cuda_time);
}
```

**Expected:** CUDA should be 10-100x faster for large tensors.

---

## Next Implementation Steps

Your current stack:
- ✅ Storage abstraction (CPU/CUDA malloc)
- ✅ Tensor views (zero-copy transpose via strides)
- ✅ CUDA kernel infrastructure (fill)
- ❌ matmul (needs cuBLAS)
- ❌ Attention kernel
- ❌ Autograd CUDA support

### Immediate TODO:

1. **Implement cuBLAS matmul** (Tensor.cu:86)
2. **Add elementwise CUDA kernels** (add, mul, gelu)
3. **Port attention to CUDA**
4. **Profile with Nsight Compute**

---

## Debugging CUDA Kernels

### Enable CUDA Debugging

```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
cuda-gdb ./test_cuda_basic
```

### Check for Memory Leaks

```bash
# Requires CUDA toolkit
compute-sanitizer ./test_cuda_basic
```

### Profile Kernels

```bash
# Install NVIDIA Nsight Compute
ncu --set full -o profile ./test_cuda_basic
```

---

## Architecture Overview

```
Tensor (tensor.h)
├── Storage (storage.h)
│   ├── CPU:  malloc/free
│   └── CUDA: cudaMalloc/cudaFree
├── Shape metadata (std::vector<int>)
└── Strides (for zero-copy views)

Operations (Tensor.cu)
├── fill() → fill_kernel<<<>>> (CUDA) or std::fill (CPU)
├── clone() → cudaMemcpy (CUDA) or memcpy (CPU)
└── matmul() → cuBLAS (TODO)
```

**Key insight:** Your `data()` pointer is **polymorphic**:
- On CPU tensors: safe to dereference
- On CUDA tensors: **device pointer** - can only be used in kernels or cudaMemcpy

---

## Questions?

If you see:
- Segfault → You're trying to read GPU memory from CPU
- `cudaErrorInvalidConfiguration` → Kernel launch config is wrong
- `cudaErrorLaunchFailure` → Kernel crashed (use cuda-gdb)

**Good luck building production-grade GPU transformers!**
