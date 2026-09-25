#pragma once
#include "grad/data/dataset.h"
#include "grad/transformer/tensor.h"
#include <memory>
#include <random>

struct Batch {
    Tensor input;
    Tensor target;
    
    Batch(int batch_size, int seq_length)
        : input(batch_size, seq_length, 1),
          target(batch_size, seq_length, 1) {}
};

// Shuffled loading draws each row's window uniformly at random (sampling
// with replacement across an epoch) instead of walking a materialized
// permutation. A permutation costs 8 bytes per window - 1.4GB for a
// TinyStories-sized mapped corpus - and a run that consumes a fraction of
// an epoch can't statistically distinguish the two. Sequential (shuffle
// off) iteration is exact and unchanged.
class DataLoader {
private:
    std::shared_ptr<Dataset> dataset_;
    int batch_size_;
    bool shuffle_;
    size_t current_index_;
    std::mt19937 rng_;

public:
    DataLoader(std::shared_ptr<Dataset> dataset, int batch_size, bool shuffle = true, unsigned int seed = 42);
    
    [[nodiscard]] bool has_next() const;
    [[nodiscard]] Batch next_batch();
    void reset();
    
    size_t num_batches() const {
        return (dataset_->size() + batch_size_ - 1) / batch_size_;
    }
    
    size_t dataset_size() const {
        return dataset_->size();
    }

    const std::shared_ptr<Dataset>& dataset() const { return dataset_; }
    int batch_size() const { return batch_size_; }
};