#include "data/token_file.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace {

constexpr char kMagic[4] = {'T', 'O', 'K', '1'};
constexpr size_t kHeaderBytes = 4 + sizeof(uint32_t) + sizeof(uint64_t);

}  // namespace

namespace tokenfile {

void write(const std::string& path, const std::vector<int>& tokens, int vocab_size) {
    if (vocab_size <= 0 || vocab_size > 65536) {
        throw std::invalid_argument("token file requires 0 < vocab_size <= 65536 (uint16 storage)");
    }

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open for writing: " + path);
    }

    const uint32_t vocab = static_cast<uint32_t>(vocab_size);
    const uint64_t count = tokens.size();
    file.write(kMagic, 4);
    file.write(reinterpret_cast<const char*>(&vocab), sizeof(vocab));
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));

    // Convert and write in chunks so huge corpora never need a full-size
    // duplicate buffer in memory.
    constexpr size_t kChunk = 1 << 20;
    std::vector<uint16_t> buffer(std::min(kChunk, tokens.size()));
    size_t written = 0;
    while (written < tokens.size()) {
        size_t n = std::min(kChunk, tokens.size() - written);
        for (size_t i = 0; i < n; i++) {
            int t = tokens[written + i];
            if (t < 0 || t >= vocab_size) {
                throw std::runtime_error("token id out of range for vocab while writing " + path);
            }
            buffer[i] = static_cast<uint16_t>(t);
        }
        file.write(reinterpret_cast<const char*>(buffer.data()), n * sizeof(uint16_t));
        written += n;
    }
    if (!file.good()) {
        throw std::runtime_error("Write failed: " + path);
    }
}

bool exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

}  // namespace tokenfile

MappedTokenDataset::MappedTokenDataset(const std::string& path, int seq_length, int stride)
    : seq_length_(seq_length), stride_(stride) {
    if (stride < 1) {
        throw std::invalid_argument("stride must be >= 1");
    }

    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("Cannot open token file: " + path);
    }

    struct stat st;
    if (::fstat(fd_, &st) != 0 || static_cast<size_t>(st.st_size) < kHeaderBytes) {
        ::close(fd_);
        throw std::runtime_error("Token file truncated or unreadable: " + path);
    }
    map_bytes_ = static_cast<size_t>(st.st_size);

    map_ = ::mmap(nullptr, map_bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (map_ == MAP_FAILED) {
        ::close(fd_);
        throw std::runtime_error("mmap failed for token file: " + path);
    }

    const char* base = static_cast<const char*>(map_);
    if (std::memcmp(base, kMagic, 4) != 0) {
        throw std::runtime_error("Not a token file (bad magic): " + path);
    }
    uint32_t vocab;
    uint64_t count;
    std::memcpy(&vocab, base + 4, sizeof(vocab));
    std::memcpy(&count, base + 4 + sizeof(vocab), sizeof(count));
    vocab_size_ = static_cast<int>(vocab);
    count_ = static_cast<size_t>(count);

    if (map_bytes_ < kHeaderBytes + count_ * sizeof(uint16_t)) {
        throw std::runtime_error("Token file shorter than its header claims: " + path);
    }
    if (count_ < static_cast<size_t>(seq_length_ + 1)) {
        throw std::runtime_error("Not enough tokens for even one sequence: " + path);
    }
    tokens_ = reinterpret_cast<const uint16_t*>(base + kHeaderBytes);
}

MappedTokenDataset::~MappedTokenDataset() {
    if (map_ != nullptr && map_ != MAP_FAILED) {
        ::munmap(map_, map_bytes_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

size_t MappedTokenDataset::size() const {
    return (count_ - seq_length_ - 1) / stride_ + 1;
}

std::pair<std::vector<int>, std::vector<int>> MappedTokenDataset::get_item(size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("MappedTokenDataset index out of range");
    }
    const size_t start = index * stride_;

    std::vector<int> input(seq_length_);
    std::vector<int> target(seq_length_);
    for (int i = 0; i < seq_length_; i++) {
        input[i] = tokens_[start + i];
        target[i] = tokens_[start + 1 + i];
    }
    return {input, target};
}
