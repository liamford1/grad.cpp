#include "grad/data/dataset.h"
#include <algorithm>
#include <stdexcept>
#include <utility>

TextDataset::TextDataset(const std::vector<int>& tokens, int seq_length, int stride)
    : token_ids_(tokens), seq_length_(seq_length), stride_(stride) {
    if (tokens.size() < static_cast<size_t>(seq_length + 1)) {
        throw std::runtime_error("Not enough tokens for even one sequence");
    }
    if (stride < 1) {
        throw std::invalid_argument("stride must be >= 1");
    }
}

size_t TextDataset::size() const {
    return (token_ids_.size() - seq_length_ - 1) / stride_ + 1;
}

std::pair<std::vector<int>, std::vector<int>> TextDataset::get_item(size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("Dataset index out of range");
    }
    const size_t start = index * stride_;

    std::vector<int> input(seq_length_);
    for (int i = 0; i < seq_length_; i++) {
        input[i] = token_ids_[start + i];
    }

    std::vector<int> target(seq_length_);
    for (int i = 0; i < seq_length_; i++) {
        target[i] = token_ids_[start + 1 + i];
    }
    return {input, target};
}

SpreadSubset::SpreadSubset(std::shared_ptr<const Dataset> source, size_t count)
    : source_(std::move(source)), count_(0) {
    if (!source_) {
        throw std::invalid_argument("SpreadSubset needs a source dataset");
    }
    count_ = std::min(count, source_->size());
}

std::pair<std::vector<int>, std::vector<int>> SpreadSubset::get_item(size_t index) const {
    if (index >= count_) {
        throw std::out_of_range("SpreadSubset index out of range");
    }
    // floor(index * n / count), split so no intermediate exceeds
    // count * count: exact without overflow for any count below 2^32.
    const size_t n = source_->size();
    const size_t source_index = index * (n / count_) + index * (n % count_) / count_;
    return source_->get_item(source_index);
}
