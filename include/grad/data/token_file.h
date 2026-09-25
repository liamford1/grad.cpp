#pragma once
#include "grad/data/dataset.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Pre-tokenized corpus files: encode once with `prepare`, then train from
// the binary directly. Format: "TOK1" magic, uint32 vocab_size,
// uint64 token count, then uint16 token ids (so vocab must fit in 65536).
namespace tokenfile {

// Writes count tokens starting at data. Pointer + count (rather than a
// vector) so a large corpus can be split into train/val files without
// materializing sub-vector copies.
void write(const std::string& path, const int* data, size_t count, int vocab_size);

inline void write(const std::string& path, const std::vector<int>& tokens, int vocab_size) {
    write(path, tokens.data(), tokens.size(), vocab_size);
}

[[nodiscard]] bool exists(const std::string& path);

// unique_ptr deleter for a read-only file mapping. Declared at namespace
// scope rather than nested in MappedTokenDataset: GCC does not treat a
// nested class with a default member initializer as default-constructible
// until the enclosing class is complete, and unique_ptr requires that.
struct Unmap {
    size_t bytes = 0;
    void operator()(void* addr) const noexcept;
};

}  // namespace tokenfile

// Dataset backed by a memory-mapped token file. The kernel pages in only
// the windows actually read, so usable corpus size is bounded by disk, not
// RAM, and training starts without re-encoding anything.
//
// Token ids are checked against the header's vocab as each window is read
// (a compare per token, next to a full forward pass) rather than by a scan
// at open, which would page in the whole file: 726MB for TinyStories.
class MappedTokenDataset : public Dataset {
    private:
        // munmap on destruction. The file descriptor is closed as soon as
        // the mapping exists; the mapping keeps the file alive by itself.
        std::unique_ptr<void, tokenfile::Unmap> map_;
        const uint16_t* tokens_ = nullptr;
        size_t count_ = 0;
        int vocab_size_ = 0;
        int seq_length_;
        int stride_;
    public:
        // stride = 1 for training windows, seq_length for non-overlapping
        // evaluation windows (same convention as TextDataset).
        MappedTokenDataset(const std::string& path, int seq_length, int stride = 1);
        MappedTokenDataset(const MappedTokenDataset&) = delete;
        MappedTokenDataset& operator=(const MappedTokenDataset&) = delete;

        size_t size() const override;
        std::pair<std::vector<int>, std::vector<int>> get_item(size_t index) const override;

        int vocabSize() const { return vocab_size_; }
        size_t tokenCount() const { return count_; }
};
