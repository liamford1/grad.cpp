#pragma once

// The pre-tokenizer of the byte-level BPE (ByteBpe): a hand-written matcher
// for GPT-2's pattern
//
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// It splits text into pieces that BPE merges never cross. \p{L} and \p{N}
// are Unicode general categories L* and N* (tables generated from Unicode
// 16.0 by tools/gen_unicode_tables.py) and \s is the White_Space property.
// Text is decoded as UTF-8; each byte of an invalid sequence counts as one
// code point of class Other, as U+FFFD would. The pieces always partition
// the input: concatenated, they give back every byte.

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace grad::pretok {

// Identifies these rules (and the Unicode tables) in tokenizer files and
// fingerprints. Any change to what next_boundary returns needs a new id.
inline constexpr std::uint32_t kGpt2Unicode16 = 1;

enum class CharClass : std::uint8_t { Letter, Number, Space, Other };

struct Codepoint {
    std::uint32_t value;   // the code point; for an invalid byte, the byte
    std::uint32_t length;  // bytes it occupies, 1 to 4
    CharClass cls;
};

// Class of a Unicode scalar value.
[[nodiscard]] CharClass classify(std::uint32_t cp);

// Decodes the code point that starts at text[pos]; pos < text.size().
[[nodiscard]] Codepoint decode_at(std::string_view text, std::size_t pos);

// End (exclusive) of the pre-token that starts at pos; pos < text.size().
[[nodiscard]] std::size_t next_boundary(std::string_view text, std::size_t pos);

// Every pre-token of text, in order.
[[nodiscard]] std::vector<std::string_view> split(std::string_view text);

}  // namespace grad::pretok
