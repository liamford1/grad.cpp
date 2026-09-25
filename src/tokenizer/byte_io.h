#pragma once

// Internal helpers for the tokenizer sources: little-endian fixed-width
// fields written and read with shifts (so a file means the same on every
// host, whatever its byte order), and the FNV-1a hash behind tokenizer
// fingerprints.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace grad::byte_io {

inline void put_u32(std::string& out, std::uint32_t v) {
    for (int shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<char>((v >> shift) & 0xFFu));
    }
}

inline void put_u64(std::string& out, std::uint64_t v) {
    for (int shift = 0; shift < 64; shift += 8) {
        out.push_back(static_cast<char>((v >> shift) & 0xFFu));
    }
}

// Bounds-checked reads from an in-memory file image. Every failure throws
// std::runtime_error naming the field and the file.
class Reader {
public:
    Reader(std::string_view data, std::string path) : data_(data), path_(std::move(path)) {}

    std::uint32_t u32(const char* what) { return static_cast<std::uint32_t>(fixed(4, what)); }
    std::uint64_t u64(const char* what) { return fixed(8, what); }

    std::string_view bytes(std::size_t n, const char* what) {
        need(n, what);
        const std::string_view out = data_.substr(pos_, n);
        pos_ += n;
        return out;
    }

    [[nodiscard]] bool at_end() const { return pos_ == data_.size(); }
    [[nodiscard]] std::size_t remaining() const { return data_.size() - pos_; }

    [[noreturn]] void fail(const std::string& message) const {
        throw std::runtime_error("Tokenizer file " + path_ + ": " + message);
    }

private:
    void need(std::size_t n, const char* what) const {
        if (data_.size() - pos_ < n) fail(std::string("truncated reading ") + what);
    }

    std::uint64_t fixed(std::size_t n, const char* what) {
        need(n, what);
        std::uint64_t v = 0;
        for (std::size_t i = 0; i < n; i++) {
            v |= static_cast<std::uint64_t>(static_cast<unsigned char>(data_[pos_ + i])) << (8 * i);
        }
        pos_ += n;
        return v;
    }

    std::string_view data_;
    std::string path_;
    std::size_t pos_ = 0;
};

// 64-bit FNV-1a. Fed only with fixed-width little-endian fields and
// length-prefixed byte strings, so the value is the same everywhere.
class Fnv1a64 {
public:
    void bytes(std::string_view data) {
        for (const char c : data) {
            hash_ = (hash_ ^ static_cast<unsigned char>(c)) * 0x100000001b3ULL;
        }
    }
    void u32(std::uint32_t v) {
        std::string buf;
        put_u32(buf, v);
        bytes(buf);
    }
    // A length-prefixed string, so ("ab", "c") and ("a", "bc") differ.
    void str(std::string_view s) {
        u32(static_cast<std::uint32_t>(s.size()));
        bytes(s);
    }
    [[nodiscard]] std::uint64_t value() const { return hash_; }

private:
    std::uint64_t hash_ = 0xcbf29ce484222325ULL;
};

}  // namespace grad::byte_io
