#include "grad/data/dataloader.h"
#include <algorithm>
#include <stdexcept>
#include <vector>

DataLoader::DataLoader(std::shared_ptr<Dataset> dataset, int batch_size, bool shuffle, unsigned int seed)
    : dataset_(dataset),
      batch_size_(batch_size),
      shuffle_(shuffle),
      current_index_(0),
      rng_(seed) {}

bool DataLoader::has_next() const {
    return current_index_ < dataset_->size();
}

Batch DataLoader::next_batch() {
    if (!has_next()) {
        throw std::runtime_error("No more batches available. Call reset() to start new epoch.");
    }

    size_t remaining = dataset_->size() - current_index_;
    int actual_batch_size = std::min(static_cast<size_t>(batch_size_), remaining);

    std::vector<size_t> indices(actual_batch_size);
    if (shuffle_) {
        std::uniform_int_distribution<size_t> pick(0, dataset_->size() - 1);
        for (int b = 0; b < actual_batch_size; b++) {
            indices[b] = pick(rng_);
        }
    } else {
        for (int b = 0; b < actual_batch_size; b++) {
            indices[b] = current_index_ + b;
        }
    }

    auto [first_input, first_target] = dataset_->get_item(indices[0]);
    int seq_length = first_input.size();

    Batch batch(actual_batch_size, seq_length);

    for (int b = 0; b < actual_batch_size; b++) {
        auto [input, target] = dataset_->get_item(indices[b]);

        for (int s = 0; s < seq_length; s++) {
            batch.input.setValue(b, s, 0, static_cast<float>(input[s]));
            batch.target.setValue(b, s, 0, static_cast<float>(target[s]));
        }
    }
    current_index_ += actual_batch_size;
    return batch;
}

void DataLoader::reset() {
    // The RNG deliberately carries across epochs so a reshuffled "epoch"
    // draws fresh windows, matching the old permute-every-epoch behavior.
    current_index_ = 0;
}
