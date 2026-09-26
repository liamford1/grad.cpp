#pragma once

// What a checkpoint records about the tokenizer its embedding rows belong
// to. Kept apart from tokenizer.h so gpt_model.h can carry it without
// pulling in the tokenizers.

#include <cstdint>
#include <string>

namespace grad {

// Tokenizer families. The values are stored in checkpoints and tokenizer
// fingerprints: never renumber them.
enum class TokenizerKind : std::uint32_t {
    BpeV1 = 1,    // whitespace-split BPE with '_' word markers (BPETokenizer)
    ByteBpe = 2,  // lossless byte-level BPE (ByteBpe)
};

// "v1" / "v2", as --tokenizer spells them.
[[nodiscard]] const char* tokenizer_kind_flag(TokenizerKind kind);

struct TokenizerFingerprint {
    TokenizerKind kind = TokenizerKind::BpeV1;
    std::uint64_t hash = 0;

    friend bool operator==(const TokenizerFingerprint&, const TokenizerFingerprint&) = default;
};

// "v2:0123456789abcdef", for messages.
[[nodiscard]] std::string to_string(const TokenizerFingerprint& fingerprint);

}  // namespace grad
