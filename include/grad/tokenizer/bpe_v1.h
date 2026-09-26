#pragma once

// BpeV1: the original whitespace BPE (BPETokenizer) behind the Tokenizer
// interface. It encodes, decodes and saves exactly as BPETokenizer does, so
// v1 caches, token files and checkpoints keep working byte for byte:
// whitespace structure is dropped and '_' decodes as a space.

#include "grad/tokenizer/bpe_tokenizer.h"
#include "grad/tokenizer/tokenizer.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace grad {

class BpeV1 final : public Tokenizer {
public:
    explicit BpeV1(BPETokenizer tokenizer);

    // Trains a BPETokenizer of vocab_size on text (BPETokenizer::train).
    [[nodiscard]] static BpeV1 train(const std::string& text, int vocab_size);
    // Loads a v1 cache (BPETokenizer::load).
    [[nodiscard]] static BpeV1 load(const std::string& path);

    [[nodiscard]] const BPETokenizer& impl() const { return impl_; }

    [[nodiscard]] TokenizerKind kind() const override { return TokenizerKind::BpeV1; }
    [[nodiscard]] std::vector<int> encode(std::string_view text) const override;
    [[nodiscard]] std::string decode(std::span<const int> ids) const override;
    [[nodiscard]] int vocab_size() const override;
    [[nodiscard]] std::optional<int> eos_id() const override;
    [[nodiscard]] std::optional<int> special_id(std::string_view token) const override;
    [[nodiscard]] std::uint64_t fingerprint() const override { return fingerprint_; }
    void save(const std::string& path) const override;

private:
    BPETokenizer impl_;
    std::uint64_t fingerprint_;
};

}  // namespace grad
