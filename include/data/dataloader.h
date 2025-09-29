#pragma once
#include "dataset.h"
#include "../transformer/tensor.h"
#include <vector>
#include <memory>
#include <random>

struct Batch {
    Tensor input;
    Tensor target;
    
    Batch(int batch_size, int seq_length)
        : input(batch_size, seq_length),
          target(batch_size, seq_length) {}
};

class DataLoader {
private:
    std::shared_ptr<Dataset> dataset_;
    int batch_size_;
    bool shuffle_;
    std::vector<size_t> indices_;
    size_t current_index_;
    std::mt19937 rng_;
    
public:
    DataLoader(std::shared_ptr<Dataset> dataset, int batch_size, bool shuffle = true, unsigned int seed = 42);
    
    bool has_next() const;
    Batch next_batch();
    void reset();
    
    size_t num_batches() const {
        return (dataset_->size() + batch_size_ - 1) / batch_size_;
    }
    
    size_t dataset_size() const {
        return dataset_->size();
    }
};