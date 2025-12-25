#pragma once
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>
#include <iostream>

enum class Device {
    CPU,
    CUDA
};

class Storage {
    public:
        void* data_ptr;
        size_t size_bytes;
        Device device;

        Storage(size_t size, Device dev) : size_bytes(size), device(dev) {
            if (device == Device::CPU) {
                data_ptr = malloc(size);
                if (!data_ptr) throw std::runtime_error("CPU OOM");
            } else {
                cudaError_t err = cudaMallocManaged(&data_ptr, size);
                if (err != cudaSuccess) {
                    throw std::runtime_error("CUDA OOM: " + std::string(cudaGetErrorString(err)));
                }
            }
        }

        ~Storage() {
            if (device == Device::CPU) {
                free(data_ptr);
            } else {
                cudaFree(data_ptr);
            }
        }

        Storage(const Storage&) = delete;
        Storage& operator=(const Storage&) = delete;

        void check_memory(size_t required_bytes) {
            size_t free_byte, total_byte;
            cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

            if (cuda_status == cudaSuccess) {
                if (required_bytes > free_byte) {
                    throw std::runtime_error("CUDA OOM: Update your resume, you need more VRAM.");
                }
            }
        }
};