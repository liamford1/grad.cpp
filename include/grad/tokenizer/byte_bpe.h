#pragma once

// ByteBpe: tokenizer v2, a lossless byte-level BPE. See
// docs/design/tokenizer-v2.md.
//
// Ids 0-255 are the bytes, 256.. the merges in the order they were learned
// (merge r produces id 256 + r), and the special tokens follow the merges.
// decode(encode(s)) == s for every byte string, invalid UTF-8 included.
// Text is pre-tokenized with GPT-2's pattern (pretokenize.h) and merges
// never cross a pre-token. Special tokens (by default "<|endoftext|>") are
// recognized in the text by encode(), become one id each, and are never
// learned or produced by merges.

#include "grad/tokenizer/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace grad {

// ByteBpe::train's options. At namespace scope because a nested class with
// default member initializers cannot be a default argument in its
// enclosing class.
struct ByteBpeTrainOptions {
    bool progress = false;  // print a progress line to stdout
};

class ByteBpe final : public Tokenizer {
public:
    static constexpr int kByteTokens = 256;
    static constexpr std::string_view kEndOfText = "<|endoftext|>";
    static constexpr std::string_view kFileMagic = "GTOK";
    static constexpr std::uint32_t kFileVersion = 2;

    // A merge: the ids of the left and right tokens it joins.
    using Merge = std::pair<int, int>;

    using TrainOptions = ByteBpeTrainOptions;

    // The 256 byte tokens and the given specials, no merges.
    explicit ByteBpe(std::vector<std::string> specials = {std::string(kEndOfText)});
    // An explicit merge list. Throws std::invalid_argument unless every
    // merge joins bytes or earlier merges and the specials are non-empty,
    // distinct, and at most kMaxSpecialBytes long.
    ByteBpe(std::vector<Merge> merges, std::vector<std::string> specials);

    // Learns vocab_size - 256 - specials.size() merges from text (fewer if
    // the text runs out of pairs). Each step merges the pair with the
    // highest count, ties going to the lexicographically smallest (left
    // bytes, right bytes), so the result does not depend on the platform or
    // standard library. Special tokens in the text are excluded.
    [[nodiscard]] static ByteBpe train(
        std::string_view text, int vocab_size,
        std::vector<std::string> specials = {std::string(kEndOfText)},
        const TrainOptions& options = {});

    // Reads a GTOK v2 file. Throws std::runtime_error on a missing,
    // truncated, corrupt or implausible file.
    [[nodiscard]] static ByteBpe load(const std::string& path);

    // Appends a special token and returns its id; the fingerprint changes.
    int add_special_token(std::string token);

    [[nodiscard]] TokenizerKind kind() const override { return TokenizerKind::ByteBpe; }
    // Inputs over 1MB are split into chunks encoded in parallel; the
    // result is always the same as a serial encode.
    [[nodiscard]] std::vector<int> encode(std::string_view text) const override;
    [[nodiscard]] std::string decode(std::span<const int> ids) const override;
    [[nodiscard]] int vocab_size() const override { return static_cast<int>(token_bytes_.size()); }
    [[nodiscard]] std::optional<int> eos_id() const override { return special_id(kEndOfText); }
    [[nodiscard]] std::optional<int> special_id(std::string_view token) const override;
    [[nodiscard]] std::uint64_t fingerprint() const override { return fingerprint_; }
    void save(const std::string& path) const override;

    // Encodes special-token text as ordinary bytes.
    [[nodiscard]] std::vector<int> encode_ordinary(std::string_view text) const;
    // encode() with chunks of about chunk_bytes (0: a single chunk). Every
    // chunk size gives the same ids.
    [[nodiscard]] std::vector<int> encode_chunked(std::string_view text,
                                                  std::size_t chunk_bytes) const;

    [[nodiscard]] const std::vector<Merge>& merges() const { return merges_; }
    [[nodiscard]] const std::vector<std::string>& specials() const { return specials_; }
    // The bytes id decodes to. Throws std::out_of_range for an unknown id.
    [[nodiscard]] const std::string& token_bytes(int id) const;

    static constexpr std::size_t kMaxSpecialBytes = 1024;

private:
    struct EncodeCache;

    // Derives token_bytes_, ranks_ and fingerprint_ from merges_ and
    // specials_, validating both.
    void rebuild();
    [[nodiscard]] int merge_rank(int left, int right) const;
    void encode_piece(std::string_view piece, std::vector<int>& out) const;
    void encode_ordinary_into(std::string_view text, EncodeCache& cache,
                              std::vector<int>& out) const;
    void encode_into(std::string_view text, bool with_specials, std::vector<int>& out) const;

    std::vector<Merge> merges_;
    std::vector<std::string> specials_;
    std::vector<std::string> token_bytes_;          // id -> bytes
    std::unordered_map<std::uint64_t, int> ranks_;  // (left, right) -> merge index
    std::uint64_t fingerprint_ = 0;
};

}  // namespace grad
