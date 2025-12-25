#pragma once
#include "storage.h"
#include <vector>
#include <memory>
#include <initializer_list>
#include <algorithm>
#include <stdexcept>

class Tensor {
    private:
        std::shared_ptr<Storage> storage;
        size_t storage_offset;
        std::vector<int> shape;
        std::vector<int> strides;

        static std::vector<int> calculate_contiguous_strides(const std::vector<int>& shape) {
            std::vector<int> strides(shape.size());
            int stride = 1;
            for (int i = shape.size() - 1; i >= 0; --i) {
                strides[i] = stride;
                stride *= shape[i];
            }
            return strides;
        }

    public:
        Tensor(std::vector<int> shape, Device device = Device::CPU);

        float* data() {
            return static_cast<float*>(storage->data_ptr) + storage_offset;
        }

        const float* data() const {
            return static_cast<const float*>(storage->data_ptr) + storage_offset;
        }
        
        Tensor transpose(int dim0, int dim1) const {
        Tensor result = *this;
        std::swap(result.shape[dim0], result.shape[dim1]);
        std::swap(result.strides[dim0], result.strides[dim1]);
            return result;
        }

        Tensor matmul(const Tensor& other) const;
        void fill(float value);
        Tensor clone() const;

        const std::vector<int>& getShape() const { return shape; }
        const std::vector<int>& getStrides() const { return strides; }
        Device getDevice() const { return storage->device; }

        size_t numel() const {
            size_t n = 1;
            for(int s : shape) n *= s;
            return n;
        }

        Tensor(size_t rows, size_t cols, Device device = Device::CPU)
            : Tensor(std::vector<int>{(int)rows, (int)cols}, device) {}

        Tensor(size_t batch, size_t rows, size_t cols, Device device = Device::CPU)
            : Tensor(std::vector<int>{(int)batch, (int)rows, (int)cols}, device) {}

        float getValue(int row, int col) const;
        void setValue(int row, int col, float value);

        float* raw() { return data(); }
        size_t getRows() const { return shape.size() > 1 ? shape[shape.size()-2] : 1; }
        size_t getCols() const { return shape.empty() ? 0 : shape.back(); }
        size_t getBatchSize() const { return shape.size() > 2 ? shape[0] : 1; }
};
