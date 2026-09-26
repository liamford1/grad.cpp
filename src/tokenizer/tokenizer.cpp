#include "grad/tokenizer/tokenizer.h"

#include "grad/tokenizer/bpe_v1.h"
#include "grad/tokenizer/byte_bpe.h"

#include "byte_io.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace grad {

const char* tokenizer_kind_flag(TokenizerKind kind) {
    switch (kind) {
        case TokenizerKind::BpeV1:
            return "v1";
        case TokenizerKind::ByteBpe:
            return "v2";
    }
    return "unknown";
}

std::string to_string(const TokenizerFingerprint& fingerprint) {
    std::array<char, 17> hex{};
    std::snprintf(hex.data(), hex.size(), "%016llx",
                  static_cast<unsigned long long>(fingerprint.hash));
    return std::string(tokenizer_kind_flag(fingerprint.kind)) + ":" + hex.data();
}

namespace {

// Canonical v1 identity: the id-ordered token table and the merge list.
// The cache stores the vocabulary in hash-map order, so the table is
// sorted by id first. Frozen: checkpoints record this value.
std::uint64_t v1_fingerprint(const BPETokenizer& tokenizer) {
    std::vector<std::pair<int, const std::string*>> tokens;
    tokens.reserve(tokenizer.getIdToToken().size());
    for (const auto& [id, token] : tokenizer.getIdToToken()) tokens.emplace_back(id, &token);
    std::sort(tokens.begin(), tokens.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    byte_io::Fnv1a64 hash;
    hash.str("grad.tokenizer.bpe_v1");
    hash.u32(static_cast<std::uint32_t>(tokens.size()));
    for (const auto& [id, token] : tokens) {
        hash.u32(static_cast<std::uint32_t>(id));
        hash.str(*token);
    }
    const auto& merges = tokenizer.getMerges();
    hash.u32(static_cast<std::uint32_t>(merges.size()));
    for (const auto& [left, right] : merges) {
        hash.str(left);
        hash.str(right);
    }
    return hash.value();
}

}  // namespace

BpeV1::BpeV1(BPETokenizer tokenizer)
    : impl_(std::move(tokenizer)), fingerprint_(v1_fingerprint(impl_)) {}

BpeV1 BpeV1::train(const std::string& text, int vocab_size) {
    BPETokenizer tokenizer(vocab_size);
    tokenizer.train(text);
    return BpeV1(std::move(tokenizer));
}

BpeV1 BpeV1::load(const std::string& path) {
    // The constructor's vocab size only steers train(); load() replaces
    // the whole vocabulary with the file's.
    BPETokenizer tokenizer(0);
    tokenizer.load(path);
    return BpeV1(std::move(tokenizer));
}

std::vector<int> BpeV1::encode(std::string_view text) const {
    return impl_.encode(text);
}

std::string BpeV1::decode(std::span<const int> ids) const {
    return impl_.decode(std::vector<int>(ids.begin(), ids.end()));
}

int BpeV1::vocab_size() const {
    return impl_.getCurrentVocabSize();
}

std::optional<int> BpeV1::eos_id() const {
    return special_id("<eos>");
}

std::optional<int> BpeV1::special_id(std::string_view token) const {
    if (token != "<pad>" && token != "<eos>" && token != "<unk>") return std::nullopt;
    const int id = impl_.findToken(std::string(token));
    if (id < 0) return std::nullopt;
    return id;
}

void BpeV1::save(const std::string& path) const {
    impl_.save(path);
}

TokenizerKind detect_tokenizer_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open tokenizer file: " + path);
    std::array<char, 4> magic{};
    file.read(magic.data(), magic.size());
    if (file.gcount() == 4 && std::string_view(magic.data(), 4) == ByteBpe::kFileMagic) {
        return TokenizerKind::ByteBpe;
    }
    return TokenizerKind::BpeV1;
}

std::unique_ptr<Tokenizer> load_tokenizer(const std::string& path) {
    if (detect_tokenizer_file(path) == TokenizerKind::ByteBpe) {
        return std::make_unique<ByteBpe>(ByteBpe::load(path));
    }
    return std::make_unique<BpeV1>(BpeV1::load(path));
}

}  // namespace grad
