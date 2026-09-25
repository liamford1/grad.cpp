#include "grad/tokenizer/pretokenize.h"

#include "unicode_tables.h"

#include <algorithm>
#include <array>
#include <span>

namespace grad::pretok {

namespace {

constexpr std::array<CharClass, 128> make_ascii_classes() {
    std::array<CharClass, 128> classes{};
    for (std::size_t c = 0; c < classes.size(); c++) {
        CharClass cls = CharClass::Other;
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
            cls = CharClass::Letter;
        } else if (c >= '0' && c <= '9') {
            cls = CharClass::Number;
        } else if (c == ' ' || (c >= 0x09 && c <= 0x0D)) {
            // \t \n \v \f \r and space: ASCII's White_Space code points.
            // 0x1C-0x1F are not White_Space (Python's str.isspace
            // disagrees; the regex module and GPT-2's tokenizers do not).
            cls = CharClass::Space;
        }
        classes[c] = cls;
    }
    return classes;
}

constexpr std::array<CharClass, 128> kAsciiClasses = make_ascii_classes();

bool in_ranges(std::span<const unicode::CodepointRange> ranges, std::uint32_t cp) {
    // First range whose hi >= cp; cp is in it iff lo <= cp.
    const auto it = std::lower_bound(
        ranges.begin(), ranges.end(), cp,
        [](const unicode::CodepointRange& r, std::uint32_t value) { return r.hi < value; });
    return it != ranges.end() && it->lo <= cp;
}

bool is_white_space(std::uint32_t cp) {
    if (cp < 0x80) return kAsciiClasses[cp] == CharClass::Space;
    return cp == 0x85 || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A)
           || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

bool is_continuation(unsigned char b) {
    return (b & 0xC0) == 0x80;
}

// Strict RFC 3629 decoding: the second byte's range excludes overlongs
// (E0, F0), surrogates (ED) and values past U+10FFFF (F4).
Codepoint decode(std::string_view text, std::size_t pos) {
    const auto byte = [&](std::size_t k) { return static_cast<unsigned char>(text[pos + k]); };
    const unsigned char b0 = byte(0);
    if (b0 < 0x80) return {b0, 1, kAsciiClasses[b0]};

    const Codepoint invalid{b0, 1, CharClass::Other};
    const std::size_t left = text.size() - pos;
    std::uint32_t length = 0;
    unsigned char lo = 0x80;
    unsigned char hi = 0xBF;
    if (b0 >= 0xC2 && b0 <= 0xDF) {
        length = 2;
    } else if (b0 >= 0xE0 && b0 <= 0xEF) {
        length = 3;
        if (b0 == 0xE0) lo = 0xA0;
        if (b0 == 0xED) hi = 0x9F;
    } else if (b0 >= 0xF0 && b0 <= 0xF4) {
        length = 4;
        if (b0 == 0xF0) lo = 0x90;
        if (b0 == 0xF4) hi = 0x8F;
    } else {
        return invalid;
    }
    if (left < length) return invalid;
    const unsigned char b1 = byte(1);
    if (b1 < lo || b1 > hi) return invalid;
    std::uint32_t cp = b0 & (0xFFu >> (length + 1));
    cp = (cp << 6) | (b1 & 0x3Fu);
    for (std::size_t k = 2; k < length; k++) {
        const unsigned char b = byte(k);
        if (!is_continuation(b)) return invalid;
        cp = (cp << 6) | (b & 0x3Fu);
    }
    return {cp, length, classify(cp)};
}

}  // namespace

CharClass classify(std::uint32_t cp) {
    if (cp < 0x80) return kAsciiClasses[cp];
    if (is_white_space(cp)) return CharClass::Space;
    if (in_ranges(unicode::kLetterRanges, cp)) return CharClass::Letter;
    if (in_ranges(unicode::kNumberRanges, cp)) return CharClass::Number;
    return CharClass::Other;
}

Codepoint decode_at(std::string_view text, std::size_t pos) {
    return decode(text, pos);
}

std::size_t next_boundary(std::string_view text, std::size_t pos) {
    const std::size_t n = text.size();
    const char first = text[pos];

    // 's 't 're 've 'm 'll 'd (case-sensitive, as in GPT-2).
    if (first == '\'' && pos + 1 < n) {
        const char c1 = text[pos + 1];
        if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return pos + 2;
        if (pos + 2 < n) {
            const char c2 = text[pos + 2];
            if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                return pos + 3;
            }
        }
    }

    // ` ?\p{L}+`, ` ?\p{N}+`, ` ?[^\s\p{L}\p{N}]+`: an optional leading
    // space, then a run of one class.
    Codepoint cp = decode(text, pos);
    std::size_t run = pos;
    if (first == ' ' && pos + 1 < n) {
        const Codepoint next = decode(text, pos + 1);
        if (next.cls != CharClass::Space) {
            cp = next;
            run = pos + 1;
        }
    }
    if (cp.cls != CharClass::Space) {
        std::size_t end = run + cp.length;
        while (end < n) {
            const Codepoint c = decode(text, end);
            if (c.cls != cp.cls) break;
            end += c.length;
        }
        return end;
    }

    // `\s+(?!\S)` then `\s+`: a whitespace run that reaches the end of the
    // text is one token. One followed by non-whitespace leaves its last code
    // point to start the next token (so " x" keeps its space), unless the
    // run is that single code point.
    std::size_t end = pos;
    std::size_t last = pos;
    std::size_t count = 0;
    while (end < n) {
        const Codepoint c = decode(text, end);
        if (c.cls != CharClass::Space) break;
        last = end;
        end += c.length;
        count++;
    }
    if (end == n || count == 1) return end;
    return last;
}

std::vector<std::string_view> split(std::string_view text) {
    std::vector<std::string_view> pieces;
    for (std::size_t pos = 0; pos < text.size();) {
        const std::size_t end = next_boundary(text, pos);
        pieces.push_back(text.substr(pos, end - pos));
        pos = end;
    }
    return pieces;
}

}  // namespace grad::pretok
