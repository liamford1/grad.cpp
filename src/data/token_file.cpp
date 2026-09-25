#include "grad/data/token_file.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace grad {

namespace {

constexpr char kMagic[4] = {'T', 'O', 'K', '1'};
constexpr size_t kHeaderBytes = 4 + sizeof(uint32_t) + sizeof(uint64_t);

// MAP_FAILED is ((void *) -1) in the system headers. Some GCC versions
// report -Wold-style-cast through the macro's expansion, so the one
// comparison against it is fenced off here.
bool map_failed(const void* addr) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
    return addr == MAP_FAILED;
#pragma GCC diagnostic pop
}

}  // namespace

namespace tokenfile {

void write(const std::string& path, const int* data, size_t count, int vocab_size) {
    if (vocab_size <= 0 || vocab_size > 65536) {
        throw std::invalid_argument("token file requires 0 < vocab_size <= 65536 (uint16 storage)");
    }

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open for writing: " + path);
    }

    const uint32_t vocab = static_cast<uint32_t>(vocab_size);
    const uint64_t count64 = count;
    file.write(kMagic, 4);
    file.write(reinterpret_cast<const char*>(&vocab), sizeof(vocab));
    file.write(reinterpret_cast<const char*>(&count64), sizeof(count64));

    // Convert and write in chunks so huge corpora never need a full-size
    // duplicate buffer in memory.
    constexpr size_t kChunk = 1 << 20;
    std::vector<uint16_t> buffer(std::min(kChunk, count));
    size_t written = 0;
    while (written < count) {
        size_t n = std::min(kChunk, count - written);
        for (size_t i = 0; i < n; i++) {
            int t = data[written + i];
            if (t < 0 || t >= vocab_size) {
                throw std::runtime_error("token id out of range for vocab while writing " + path);
            }
            buffer[i] = static_cast<uint16_t>(t);
        }
        file.write(reinterpret_cast<const char*>(buffer.data()),
                   static_cast<std::streamsize>(n * sizeof(uint16_t)));
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

void tokenfile::Unmap::operator()(void* addr) const noexcept {
    ::munmap(addr, bytes);
}

MappedTokenDataset::MappedTokenDataset(const std::string& path, int seq_length, int stride) {
    if (stride < 1) {
        throw std::invalid_argument("stride must be >= 1");
    }
    if (seq_length < 1) {
        throw std::invalid_argument("seq_length must be >= 1");
    }
    seq_length_ = static_cast<size_t>(seq_length);
    stride_ = static_cast<size_t>(stride);

    // Closes the descriptor on every exit from the constructor, including
    // the throws below; the mapping outlives it.
    struct Fd {
        int fd;
        ~Fd() {
            if (fd >= 0) ::close(fd);
        }
    } file{::open(path.c_str(), O_RDONLY)};
    if (file.fd < 0) {
        throw std::runtime_error("Cannot open token file: " + path);
    }

    struct stat st;
    if (::fstat(file.fd, &st) != 0 || st.st_size < 0
        || static_cast<size_t>(st.st_size) < kHeaderBytes) {
        throw std::runtime_error("Token file truncated or unreadable: " + path);
    }
    const size_t map_bytes = static_cast<size_t>(st.st_size);

    void* addr = ::mmap(nullptr, map_bytes, PROT_READ, MAP_PRIVATE, file.fd, 0);
    if (map_failed(addr)) {
        throw std::runtime_error("mmap failed for token file: " + path);
    }
    // Owned from here on: any later throw unmaps through the member's
    // destructor.
    map_ = std::unique_ptr<void, tokenfile::Unmap>(addr, tokenfile::Unmap{map_bytes});

    const char* base = static_cast<const char*>(addr);
    if (std::memcmp(base, kMagic, 4) != 0) {
        throw std::runtime_error("Not a token file (bad magic): " + path);
    }
    uint32_t vocab;
    uint64_t count;
    std::memcpy(&vocab, base + 4, sizeof(vocab));
    std::memcpy(&count, base + 4 + sizeof(vocab), sizeof(count));
    if (vocab == 0 || vocab > 65536) {
        throw std::runtime_error("Token file vocab " + std::to_string(vocab)
                                 + " outside (0, 65536]: " + path);
    }
    // Compared as a token count so a corrupt count cannot overflow the
    // byte-size product.
    if (count > (map_bytes - kHeaderBytes) / sizeof(uint16_t)) {
        throw std::runtime_error("Token file shorter than its header claims: " + path);
    }
    vocab_size_ = static_cast<int>(vocab);
    count_ = static_cast<size_t>(count);

    if (count_ < seq_length_ + 1) {
        throw std::runtime_error("Not enough tokens for even one sequence: " + path);
    }
    tokens_ = reinterpret_cast<const uint16_t*>(base + kHeaderBytes);
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
    // The window spans tokens [start, start + seq_length] inclusive (the
    // target is the input shifted by one).
    const uint16_t* window = tokens_ + start;
    for (size_t i = 0; i <= seq_length_; i++) {
        if (window[i] >= vocab_size_) {
            throw std::runtime_error(
                "token id " + std::to_string(window[i]) + " at offset " + std::to_string(start + i)
                + " is outside the file's vocab of " + std::to_string(vocab_size_));
        }
    }
    for (size_t i = 0; i < seq_length_; i++) {
        input[i] = window[i];
        target[i] = window[i + 1];
    }
    return {input, target};
}

}  // namespace grad
