#include "data/dataloader.h"
#include <algorithm>
#include <iostream>

DataLoader::DataLoader(std::shared_ptr<Dataset> dataset, int batch_size, bool shuffle, unsigned int seed)
    : dataset_(dataset),
      batch_size_(batch_size),
      shuffle_(shuffle),
      current_index_(0),
      rng_(seed) {
    
    indices_.resize(dataset_->size());
    for (size_t i = 0; i < dataset_->size(); i++) {
        indices_[i] = i;
    }
    
    if (shuffle_) {
        std::shuffle(indices_.begin(), indices_.end(), rng_);
    }
    
    std::cout << "Created DataLoader with " << num_batches() 
              << " batches of size " << batch_size_ << std::endl;
}

bool DataLoader::has_next() const {
    return current_index_ < indices_.size();
}

Batch DataLoader::next_batch() {
    if (!has_next()) {
        throw std::runtime_error("No more batches available. Call reset() to start new epoch.");
    }
    
    size_t remaining = indices_.size() - current_index_;
    int actual_batch_size = std::min(static_cast<size_t>(batch_size_), remaining);
    
    auto [first_input, first_target] = dataset_->get_item(indices_[current_index_]);
    int seq_length = first_input.size();
    
    Batch batch(actual_batch_size, seq_length);
    
    for (int b = 0; b < actual_batch_size; b++) {
        size_t idx = indices_[current_index_ + b];
        auto [input, target] = dataset_->get_item(idx);
        
        for (int s = 0; s < seq_length; s++) {
            batch.input.setValue(b, s, static_cast<float>(input[s]));
            batch.target.setValue(b, s, static_cast<float>(target[s]));
        }
    }
    current_index_ += actual_batch_size;
    return batch;
}

void DataLoader::reset() {
    current_index_ = 0;
    if (shuffle_) {
        std::shuffle(indices_.begin(), indices_.end(), rng_);
    }
}