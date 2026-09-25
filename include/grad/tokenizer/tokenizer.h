#pragma once

// grad::Tokenizer: the interface the trainer, the CLI and TextGen use, with
// two implementations: BpeV1 (bpe_v1.h), the original whitespace BPE that
// existing checkpoints depend on, and ByteBpe (byte_bpe.h), the lossless
// byte-level BPE. load_tokenizer() opens a file of either format.

#include "grad/tokenizer/fingerprint.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace grad {

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    [[nodiscard]] virtual TokenizerKind kind() const = 0;

    [[nodiscard]] virtual std::vector<int> encode(std::string_view text) const = 0;
    [[nodiscard]] virtual std::string decode(std::span<const int> ids) const = 0;

    // Number of token ids; every id encode returns is below it.
    [[nodiscard]] virtual int vocab_size() const = 0;
    // The end-of-text id, if the tokenizer has one.
    [[nodiscard]] virtual std::optional<int> eos_id() const = 0;
    // Id of a reserved token such as "<|endoftext|>", if it is one.
    [[nodiscard]] virtual std::optional<int> special_id(std::string_view token) const = 0;

    // Stable 64-bit hash of everything that decides how text maps to ids
    // (kind, vocabulary, merges, special tokens). Equal on every platform.
    [[nodiscard]] virtual std::uint64_t fingerprint() const = 0;
    [[nodiscard]] TokenizerFingerprint identity() const { return {kind(), fingerprint()}; }

    // Throws std::runtime_error on an I/O failure.
    virtual void save(const std::string& path) const = 0;
};

// Which format a tokenizer file is in, from its first four bytes ("GTOK"
// is v2; anything else is taken for a v1 cache). Throws if it cannot be
// opened.
[[nodiscard]] TokenizerKind detect_tokenizer_file(const std::string& path);

// Loads a tokenizer file of either format. Throws std::runtime_error on a
// missing, truncated, or corrupt file.
[[nodiscard]] std::unique_ptr<Tokenizer> load_tokenizer(const std::string& path);

}  // namespace grad
