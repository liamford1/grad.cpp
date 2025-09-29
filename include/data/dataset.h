#pragma once
#include <vector>
#include <string>
#include <memory>

class Dataset {
    public:
        virtual ~Dataset() = default;
        virtual size_t size() const = 0;
        virtual std::pair<std::vector<int>, std::vector<int>> get_item(size_t index) const = 0;
};

class TextDataset : public Dataset {
    private:
        std::vector<int> token_ids_;
        int seq_length_;
    public:
        TextDataset(const std::vector<int>& tokens, int seq_length_);

        size_t size() const override;

        std::pair<std::vector<int>, std::vector<int>> get_item(size_t index) const override;
};