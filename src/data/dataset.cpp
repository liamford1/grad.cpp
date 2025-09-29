#include "data/dataset.h"
#include <stdexcept>
#include <iostream>

TextDataset::TextDataset(const std::vector<int>& tokens, int seq_length) : token_ids_(tokens), seq_length_(seq_length) {
    if (tokens.size() < static_cast<size_t>(seq_length + 1)) {
        throw std::runtime_error("Not enough tokens for even one sequence");
    }
    std::cout << "Created TextDataset with " << tokens.size() << " tokens, seq_length=" << seq_length << std::endl;
}

size_t TextDataset::size() const {
    return token_ids_.size() - seq_length_;
}

std::pair<std::vector<int>, std::vector<int>> TextDataset::get_item(size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("Dataset index out of range");
    }
    
    std::vector<int> input(seq_length_);
    for (int i = 0; i < seq_length_; i++) {
        input[i] = token_ids_[index + i];
    }
    
    std::vector<int> target(seq_length_);
    for (int i = 0; i < seq_length_; i++) {
        target[i] = token_ids_[index + 1 + i];
    }
    return {input, target};
}