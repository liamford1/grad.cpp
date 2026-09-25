#pragma once
#include <vector>
#include <string>
#include <memory>

namespace grad {

class Dataset {
public:
    virtual ~Dataset() = default;
    virtual size_t size() const = 0;
    virtual std::pair<std::vector<int>, std::vector<int>> get_item(size_t index) const = 0;
};

class TextDataset : public Dataset {
private:
    std::vector<int> token_ids_;
    size_t seq_length_;
    size_t stride_;

public:
    // stride = 1 gives every sliding window (training); stride =
    // seq_length gives non-overlapping windows (deterministic full
    // coverage for evaluation).
    TextDataset(const std::vector<int>& tokens, int seq_length, int stride = 1);

    size_t size() const override;

    std::pair<std::vector<int>, std::vector<int>> get_item(size_t index) const override;
};

// A fixed view of `count` windows spread evenly across `source`: element i
// is source window floor(i * source.size() / count). Used for capped
// validation, where the first N windows of a contiguous split would score
// one region of the corpus (on TinyStories the first 32 val batches read
// 1.4376 while a uniform 1M-token sample reads 1.6932). The same windows
// every time, so successive evals stay comparable with each other.
class SpreadSubset : public Dataset {
private:
    std::shared_ptr<const Dataset> source_;
    size_t count_;

public:
    // count is capped at source->size().
    SpreadSubset(std::shared_ptr<const Dataset> source, size_t count);

    size_t size() const override { return count_; }
    std::pair<std::vector<int>, std::vector<int>> get_item(size_t index) const override;
};

}  // namespace grad
