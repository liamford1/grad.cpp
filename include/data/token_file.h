#pragma once
#include "dataset.h"
#include <cstdint>
#include <string>
#include <vector>

// Pre-tokenized corpus files: encode once with `prepare`, then train from
// the binary directly. Format: "TOK1" magic, uint32 vocab_size,
// uint64 token count, then uint16 token ids (so vocab must fit in 65536).
namespace tokenfile {

void write(const std::string& path, const std::vector<int>& tokens, int vocab_size);

bool exists(const std::string& path);

}  // namespace tokenfile

// Dataset backed by a memory-mapped token file. The kernel pages in only
// the windows actually read, so usable corpus size is bounded by disk, not
// RAM, and training starts without re-encoding anything.
class MappedTokenDataset : public Dataset {
    private:
        int fd_ = -1;
        void* map_ = nullptr;
        size_t map_bytes_ = 0;
        const uint16_t* tokens_ = nullptr;
        size_t count_ = 0;
        int vocab_size_ = 0;
        int seq_length_;
        int stride_;
    public:
        // stride = 1 for training windows, seq_length for non-overlapping
        // evaluation windows (same convention as TextDataset).
        MappedTokenDataset(const std::string& path, int seq_length, int stride = 1);
        ~MappedTokenDataset() override;
        MappedTokenDataset(const MappedTokenDataset&) = delete;
        MappedTokenDataset& operator=(const MappedTokenDataset&) = delete;

        size_t size() const override;
        std::pair<std::vector<int>, std::vector<int>> get_item(size_t index) const override;

        int vocabSize() const { return vocab_size_; }
        size_t tokenCount() const { return count_; }
};
