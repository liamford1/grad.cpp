#include "grad/tokenizer/byte_bpe.h"

#include "grad/tokenizer/pretokenize.h"
#include "grad/transformer/parallel.h"

#include "byte_io.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <queue>
#include <sstream>
#include <stdexcept>

namespace grad {

namespace {

// Bounds for a tokenizer file's fields: far above anything train()
// produces, low enough that a corrupt field fails with a message rather
// than as a huge allocation.
constexpr std::uint32_t kMaxVocab = 1u << 24;
constexpr std::uint32_t kMaxSpecials = 1024;

// Pieces up to this many bytes are merged by a direct scan; longer ones
// by the heap, which is O(n log n).
constexpr std::size_t kSmallPiece = 16;
// Inputs at least this large are encoded as parallel chunks.
constexpr std::size_t kParallelThreshold = std::size_t{1} << 20;
constexpr std::size_t kMinChunk = std::size_t{1} << 20;
constexpr std::size_t kMaxChunk = std::size_t{64} << 20;
// A chunk's pre-token cache starts over past this many entries, which
// bounds its memory on text with few repeats (random bytes, say).
constexpr std::size_t kCacheLimit = std::size_t{1} << 20;

constexpr std::size_t kNone = static_cast<std::size_t>(-1);

std::uint64_t pair_key(int left, int right) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32)
           | static_cast<std::uint32_t>(right);
}
int key_left(std::uint64_t key) {
    return static_cast<int>(key >> 32);
}
int key_right(std::uint64_t key) {
    return static_cast<int>(key & 0xFFFFFFFFu);
}

struct SpecialMatch {
    std::size_t pos = std::string_view::npos;
    std::size_t length = 0;
    int index = -1;  // into the specials list; -1 = none
};

// Finds special-token occurrences left to right: the earliest one at or
// after a position, the longest on a tie. Each special's next occurrence
// is cached, so a scan over the text finds each one once.
class SpecialFinder {
public:
    SpecialFinder(std::string_view text, const std::vector<std::string>& specials)
        : text_(text),
          specials_(specials),
          next_(specials.size(), 0),
          known_(specials.size(), false) {}

    SpecialMatch find(std::size_t from) {
        SpecialMatch best;
        for (std::size_t i = 0; i < specials_.size(); i++) {
            if (!known_[i] || next_[i] < from) {
                next_[i] = text_.find(specials_[i], from);
                known_[i] = true;
            }
            const std::size_t at = next_[i];
            if (at == std::string_view::npos) continue;
            if (at < best.pos || (at == best.pos && specials_[i].size() > best.length)) {
                best = {at, specials_[i].size(), static_cast<int>(i)};
            }
        }
        return best;
    }

private:
    std::string_view text_;
    const std::vector<std::string>& specials_;
    std::vector<std::size_t> next_;
    std::vector<bool> known_;
};

// Calls fn(piece) for every pre-token of text outside special tokens.
template <typename Fn>
void for_each_ordinary_piece(std::string_view text, const std::vector<std::string>& specials,
                             Fn&& fn) {
    SpecialFinder finder(text, specials);
    std::size_t pos = 0;
    while (pos <= text.size()) {
        const SpecialMatch match = finder.find(pos);
        const std::size_t end = match.index < 0 ? text.size() : match.pos;
        const std::string_view segment = text.substr(pos, end - pos);
        for (std::size_t p = 0; p < segment.size();) {
            const std::size_t q = pretok::next_boundary(segment, p);
            fn(segment.substr(p, q - p));
            p = q;
        }
        if (match.index < 0) break;
        pos = match.pos + match.length;
    }
}

// ---- Training ----------------------------------------------------------

struct Word {
    std::vector<int> symbols;
    std::int64_t freq;
};

struct HeapEntry {
    std::int64_t count;
    int left;
    int right;
};

// Heap order: highest count first, then the lexicographically smallest
// (left bytes, right bytes), compared as unsigned bytes. Two different
// tokens can spell the same bytes ("ab"+"c" and "a"+"bc"), so smaller ids
// settle what bytes cannot; the order is total, so the heap's top is
// always the same pair whatever order entries were pushed in.
struct HeapLower {
    const std::vector<std::string>* bytes;

    bool operator()(const HeapEntry& a, const HeapEntry& b) const {
        if (a.count != b.count) return a.count < b.count;
        const std::vector<std::string>& t = *bytes;
        if (const int c =
                t[static_cast<std::size_t>(a.left)].compare(t[static_cast<std::size_t>(b.left)]);
            c != 0) {
            return c > 0;
        }
        if (const int c =
                t[static_cast<std::size_t>(a.right)].compare(t[static_cast<std::size_t>(b.right)]);
            c != 0) {
            return c > 0;
        }
        if (a.left != b.left) return a.left > b.left;
        return a.right > b.right;
    }
};

std::vector<ByteBpe::Merge> learn_merges(std::string_view text, std::size_t num_merges,
                                         const std::vector<std::string>& specials,
                                         const ByteBpe::TrainOptions& options) {
    const auto start = std::chrono::steady_clock::now();

    // Unique pre-tokens with their frequencies. Single bytes have no pairs.
    std::unordered_map<std::string_view, std::int64_t> counts;
    for_each_ordinary_piece(text, specials, [&](std::string_view piece) {
        if (piece.size() >= 2) counts[piece]++;
    });
    std::vector<std::pair<std::string_view, std::int64_t>> unique(counts.begin(), counts.end());
    counts = {};
    std::sort(unique.begin(), unique.end());

    std::vector<Word> words;
    words.reserve(unique.size());
    for (const auto& [piece, freq] : unique) {
        Word word{{}, freq};
        word.symbols.reserve(piece.size());
        for (const char c : piece) word.symbols.push_back(static_cast<unsigned char>(c));
        words.push_back(std::move(word));
    }
    unique = {};

    std::unordered_map<std::uint64_t, std::int64_t> pair_counts;
    // pair -> words containing it. May hold duplicates and words that no
    // longer contain the pair; both are filtered when the pair is merged.
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> where;
    for (std::size_t w = 0; w < words.size(); w++) {
        const std::vector<int>& s = words[w].symbols;
        for (std::size_t i = 0; i + 1 < s.size(); i++) {
            const std::uint64_t key = pair_key(s[i], s[i + 1]);
            pair_counts[key] += words[w].freq;
            where[key].push_back(static_cast<std::uint32_t>(w));
        }
    }

    std::vector<std::string> bytes;
    bytes.reserve(ByteBpe::kByteTokens + num_merges);
    for (int b = 0; b < ByteBpe::kByteTokens; b++) bytes.emplace_back(1, static_cast<char>(b));

    // Lazy deletion: an entry is live only while its count is still the
    // pair's current count. Every count change pushes a fresh entry.
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapLower> heap(HeapLower{&bytes});
    for (const auto& [key, count] : pair_counts) heap.push({count, key_left(key), key_right(key)});

    std::vector<ByteBpe::Merge> merges;
    merges.reserve(num_merges);
    std::unordered_map<std::uint64_t, std::int64_t> delta;
    while (merges.size() < num_merges && !heap.empty()) {
        const HeapEntry top = heap.top();
        heap.pop();
        const std::uint64_t key = pair_key(top.left, top.right);
        const auto live = pair_counts.find(key);
        if (live == pair_counts.end() || live->second != top.count) continue;

        const int left = top.left;
        const int right = top.right;
        const int merged = ByteBpe::kByteTokens + static_cast<int>(merges.size());
        merges.emplace_back(left, right);
        bytes.push_back(bytes[static_cast<std::size_t>(left)]
                        + bytes[static_cast<std::size_t>(right)]);

        std::vector<std::uint32_t> touched = std::move(where[key]);
        where.erase(key);
        std::sort(touched.begin(), touched.end());
        touched.erase(std::unique(touched.begin(), touched.end()), touched.end());

        delta.clear();
        for (const std::uint32_t w : touched) {
            std::vector<int>& s = words[w].symbols;
            const std::int64_t freq = words[w].freq;
            bool has_pair = false;
            for (std::size_t i = 0; i + 1 < s.size() && !has_pair; i++) {
                has_pair = s[i] == left && s[i + 1] == right;
            }
            if (!has_pair) continue;

            // Net pair-count change of this word: all old pairs out, all
            // new pairs in. Pairs the merge did not touch cancel.
            for (std::size_t i = 0; i + 1 < s.size(); i++) delta[pair_key(s[i], s[i + 1])] -= freq;
            std::size_t out = 0;
            for (std::size_t i = 0; i < s.size();) {
                if (i + 1 < s.size() && s[i] == left && s[i + 1] == right) {
                    s[out++] = merged;
                    i += 2;
                } else {
                    s[out++] = s[i++];
                }
            }
            s.resize(out);
            for (std::size_t i = 0; i + 1 < s.size(); i++) {
                const std::uint64_t k = pair_key(s[i], s[i + 1]);
                delta[k] += freq;
                if (s[i] == merged || s[i + 1] == merged) where[k].push_back(w);
            }
        }

        for (const auto& [k, d] : delta) {
            if (d == 0) continue;
            const auto it = pair_counts.find(k);
            const std::int64_t count = (it == pair_counts.end() ? 0 : it->second) + d;
            if (count <= 0) {
                if (it != pair_counts.end()) pair_counts.erase(it);
                continue;
            }
            if (it == pair_counts.end()) {
                pair_counts.emplace(k, count);
            } else {
                it->second = count;
            }
            heap.push({count, key_left(k), key_right(k)});
        }
        pair_counts.erase(key);

        const std::size_t done = merges.size();
        if (options.progress && (done % 250 == 0 || done == num_merges)) {
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::ostringstream line;
            line << std::fixed << std::setprecision(1) << "\r  Training tokenizer: [" << done << "/"
                 << num_merges << "] "
                 << 100.0 * static_cast<double>(done) / static_cast<double>(num_merges)
                 << "% elapsed=" << elapsed << "s     ";
            std::cout << line.str() << std::flush;
        }
    }
    if (options.progress) std::cout << std::endl;
    return merges;
}

// ---- Chunked encoding --------------------------------------------------

// True if an occurrence of a special token starts before pos and ends
// after it, i.e. a cut at pos would split it.
bool special_straddles(std::string_view text, std::size_t pos,
                       const std::vector<std::string>& specials) {
    for (const std::string& s : specials) {
        const std::size_t len = s.size();
        if (len < 2) continue;
        const std::size_t lo = pos >= len - 1 ? pos - (len - 1) : 0;
        const std::string_view window = text.substr(lo, (pos - lo) + (len - 1));
        for (std::size_t k = window.find(s); k != std::string_view::npos;
             k = window.find(s, k + 1)) {
            if (lo + k < pos && lo + k + len > pos) return true;
        }
    }
    return false;
}

// The first chunk start at or after from: right after a '\n' that is a
// whitespace run on its own (the byte before it is ASCII and not
// whitespace) and is followed by a code point that is not whitespace, and
// not inside a special token. The pre-tokenizer makes that '\n' a token by
// itself whether or not text follows it, and starts afresh after it, so
// encoding the two sides separately gives the same ids. A longer run would
// not do: " \n" ending a chunk is one token, but two when text follows.
// text.size() if there is none.
std::size_t find_cut(std::string_view text, std::size_t from,
                     const std::vector<std::string>& specials) {
    const auto lone_newline = [&](std::size_t q) {
        if (q == 0) return true;
        const auto before = static_cast<unsigned char>(text[q - 1]);
        return before < 0x80 && pretok::classify(before) != pretok::CharClass::Space;
    };
    std::size_t q = from == 0 ? 0 : from - 1;
    while (true) {
        q = text.find('\n', q);
        if (q == std::string_view::npos || q + 1 >= text.size()) return text.size();
        const std::size_t p = q + 1;
        if (lone_newline(q) && pretok::decode_at(text, p).cls != pretok::CharClass::Space
            && !special_straddles(text, p, specials)) {
            return p;
        }
        q = p;
    }
}

// ---- File format -------------------------------------------------------

std::uint64_t compute_fingerprint(const std::vector<ByteBpe::Merge>& merges,
                                  const std::vector<std::string>& specials) {
    byte_io::Fnv1a64 hash;
    hash.str("grad.tokenizer.bytebpe");
    hash.u32(ByteBpe::kFileVersion);
    hash.u32(pretok::kGpt2Unicode16);
    hash.u32(static_cast<std::uint32_t>(specials.size()));
    for (const std::string& s : specials) hash.str(s);
    hash.u32(static_cast<std::uint32_t>(merges.size()));
    for (const auto& [left, right] : merges) {
        hash.u32(static_cast<std::uint32_t>(left));
        hash.u32(static_cast<std::uint32_t>(right));
    }
    return hash.value();
}

std::string read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open tokenizer file: " + path);
    std::string data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (file.bad()) throw std::runtime_error("Failed to read tokenizer file: " + path);
    return data;
}

}  // namespace

struct ByteBpe::EncodeCache {
    // piece -> (offset, length) of its ids in `ids`. The keys view the
    // text being encoded, which outlives the cache.
    std::unordered_map<std::string_view, std::pair<std::size_t, std::size_t>> index;
    std::vector<int> ids;
};

ByteBpe::ByteBpe(std::vector<std::string> specials) : specials_(std::move(specials)) {
    rebuild();
}

ByteBpe::ByteBpe(std::vector<Merge> merges, std::vector<std::string> specials)
    : merges_(std::move(merges)), specials_(std::move(specials)) {
    rebuild();
}

void ByteBpe::rebuild() {
    if (specials_.size() > kMaxSpecials) {
        throw std::invalid_argument("too many special tokens (" + std::to_string(specials_.size())
                                    + ")");
    }
    for (std::size_t i = 0; i < specials_.size(); i++) {
        const std::string& s = specials_[i];
        if (s.empty() || s.size() > kMaxSpecialBytes) {
            throw std::invalid_argument("special token " + std::to_string(i) + " has "
                                        + std::to_string(s.size()) + " bytes; allowed 1 to "
                                        + std::to_string(kMaxSpecialBytes));
        }
        if (std::find(specials_.begin(), specials_.begin() + static_cast<std::ptrdiff_t>(i), s)
            != specials_.begin() + static_cast<std::ptrdiff_t>(i)) {
            throw std::invalid_argument("special token '" + s + "' is listed twice");
        }
    }
    const std::size_t vocab = kByteTokens + merges_.size() + specials_.size();
    if (vocab > kMaxVocab) {
        throw std::invalid_argument("vocabulary of " + std::to_string(vocab) + " is too large");
    }

    std::vector<std::string> bytes;
    bytes.reserve(vocab);
    for (int b = 0; b < kByteTokens; b++) bytes.emplace_back(1, static_cast<char>(b));
    std::unordered_map<std::uint64_t, int> ranks;
    ranks.reserve(merges_.size());
    for (std::size_t r = 0; r < merges_.size(); r++) {
        const auto [left, right] = merges_[r];
        const auto known = static_cast<int>(kByteTokens + r);
        if (left < 0 || right < 0 || left >= known || right >= known) {
            throw std::invalid_argument("merge " + std::to_string(r) + " joins tokens "
                                        + std::to_string(left) + " and " + std::to_string(right)
                                        + ", which it does not follow");
        }
        if (!ranks.emplace(pair_key(left, right), static_cast<int>(r)).second) {
            throw std::invalid_argument("merge " + std::to_string(r) + " repeats an earlier merge");
        }
        bytes.push_back(bytes[static_cast<std::size_t>(left)]
                        + bytes[static_cast<std::size_t>(right)]);
    }
    for (const std::string& s : specials_) bytes.push_back(s);

    token_bytes_ = std::move(bytes);
    ranks_ = std::move(ranks);
    fingerprint_ = compute_fingerprint(merges_, specials_);
}

ByteBpe ByteBpe::train(std::string_view text, int vocab_size, std::vector<std::string> specials,
                       const TrainOptions& options) {
    ByteBpe tokenizer(std::move(specials));
    const long long merges = static_cast<long long>(vocab_size) - kByteTokens
                             - static_cast<long long>(tokenizer.specials_.size());
    if (merges < 0) {
        throw std::invalid_argument("vocab size " + std::to_string(vocab_size)
                                    + " leaves no room for the 256 byte tokens and "
                                    + std::to_string(tokenizer.specials_.size())
                                    + " special token(s)");
    }
    tokenizer.merges_ =
        learn_merges(text, static_cast<std::size_t>(merges), tokenizer.specials_, options);
    tokenizer.rebuild();
    return tokenizer;
}

int ByteBpe::add_special_token(std::string token) {
    specials_.push_back(std::move(token));
    try {
        rebuild();
    } catch (...) {
        specials_.pop_back();
        rebuild();
        throw;
    }
    return vocab_size() - 1;
}

std::optional<int> ByteBpe::special_id(std::string_view token) const {
    for (std::size_t i = 0; i < specials_.size(); i++) {
        if (specials_[i] == token) {
            return static_cast<int>(kByteTokens + merges_.size() + i);
        }
    }
    return std::nullopt;
}

const std::string& ByteBpe::token_bytes(int id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= token_bytes_.size()) {
        throw std::out_of_range("token id " + std::to_string(id) + " is outside the vocabulary of "
                                + std::to_string(token_bytes_.size()));
    }
    return token_bytes_[static_cast<std::size_t>(id)];
}

int ByteBpe::merge_rank(int left, int right) const {
    const auto it = ranks_.find(pair_key(left, right));
    return it == ranks_.end() ? -1 : it->second;
}

// Repeatedly merges the adjacent pair of lowest rank, leftmost first. A
// merge only creates pairs of higher rank than its own (they contain its
// new token), so this applies the merges in training order and gives the
// same tokens as training did.
void ByteBpe::encode_piece(std::string_view piece, std::vector<int>& out) const {
    const std::size_t n = piece.size();
    if (n <= kSmallPiece) {
        std::array<int, kSmallPiece> sym{};
        for (std::size_t i = 0; i < n; i++) sym[i] = static_cast<unsigned char>(piece[i]);
        std::size_t len = n;
        while (len > 1) {
            int best_rank = INT_MAX;
            std::size_t best = 0;
            for (std::size_t i = 0; i + 1 < len; i++) {
                const int r = merge_rank(sym[i], sym[i + 1]);
                if (r >= 0 && r < best_rank) {
                    best_rank = r;
                    best = i;
                }
            }
            if (best_rank == INT_MAX) break;
            sym[best] = kByteTokens + best_rank;
            std::copy(sym.begin() + static_cast<std::ptrdiff_t>(best + 2),
                      sym.begin() + static_cast<std::ptrdiff_t>(len),
                      sym.begin() + static_cast<std::ptrdiff_t>(best + 1));
            len--;
        }
        out.insert(out.end(), sym.begin(), sym.begin() + static_cast<std::ptrdiff_t>(len));
        return;
    }

    // Linked list over the bytes plus a min-heap of candidate merges keyed
    // by (rank, position). A candidate is stale once either side changed.
    struct Candidate {
        int rank;
        std::size_t pos;
        int left;
        int right;
    };
    const auto later = [](const Candidate& a, const Candidate& b) {
        return a.rank != b.rank ? a.rank > b.rank : a.pos > b.pos;
    };
    std::vector<int> id(n);
    std::vector<std::size_t> next(n);
    std::vector<std::size_t> prev(n);
    for (std::size_t i = 0; i < n; i++) {
        id[i] = static_cast<unsigned char>(piece[i]);
        next[i] = i + 1;
        prev[i] = i == 0 ? kNone : i - 1;
    }
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(later)> heap(later);
    for (std::size_t i = 0; i + 1 < n; i++) {
        const int r = merge_rank(id[i], id[i + 1]);
        if (r >= 0) heap.push({r, i, id[i], id[i + 1]});
    }
    while (!heap.empty()) {
        const Candidate c = heap.top();
        heap.pop();
        const std::size_t left = c.pos;
        if (id[left] != c.left) continue;
        const std::size_t right = next[left];
        if (right >= n || id[right] != c.right) continue;

        const int merged = kByteTokens + c.rank;
        id[left] = merged;
        id[right] = -1;
        next[left] = next[right];
        if (next[right] < n) prev[next[right]] = left;

        if (prev[left] != kNone) {
            const std::size_t p = prev[left];
            const int r = merge_rank(id[p], merged);
            if (r >= 0) heap.push({r, p, id[p], merged});
        }
        if (next[left] < n) {
            const std::size_t q = next[left];
            const int r = merge_rank(merged, id[q]);
            if (r >= 0) heap.push({r, left, merged, id[q]});
        }
    }
    for (std::size_t i = 0; i < n; i = next[i]) out.push_back(id[i]);
}

void ByteBpe::encode_ordinary_into(std::string_view text, EncodeCache& cache,
                                   std::vector<int>& out) const {
    for (std::size_t pos = 0; pos < text.size();) {
        const std::size_t end = pretok::next_boundary(text, pos);
        const std::string_view piece = text.substr(pos, end - pos);
        pos = end;
        if (piece.size() == 1) {
            out.push_back(static_cast<unsigned char>(piece[0]));
            continue;
        }
        if (const auto hit = cache.index.find(piece); hit != cache.index.end()) {
            const auto first = cache.ids.begin() + static_cast<std::ptrdiff_t>(hit->second.first);
            out.insert(out.end(), first, first + static_cast<std::ptrdiff_t>(hit->second.second));
            continue;
        }
        const std::size_t before = out.size();
        encode_piece(piece, out);
        if (cache.index.size() >= kCacheLimit) {
            cache.index.clear();
            cache.ids.clear();
        }
        cache.index.emplace(piece, std::make_pair(cache.ids.size(), out.size() - before));
        cache.ids.insert(cache.ids.end(), out.begin() + static_cast<std::ptrdiff_t>(before),
                         out.end());
    }
}

void ByteBpe::encode_into(std::string_view text, bool with_specials, std::vector<int>& out) const {
    EncodeCache cache;
    if (!with_specials || specials_.empty()) {
        encode_ordinary_into(text, cache, out);
        return;
    }
    const int first_special = static_cast<int>(kByteTokens + merges_.size());
    SpecialFinder finder(text, specials_);
    std::size_t pos = 0;
    while (true) {
        const SpecialMatch match = finder.find(pos);
        const std::size_t end = match.index < 0 ? text.size() : match.pos;
        encode_ordinary_into(text.substr(pos, end - pos), cache, out);
        if (match.index < 0) break;
        out.push_back(first_special + match.index);
        pos = match.pos + match.length;
    }
}

std::vector<int> ByteBpe::encode_ordinary(std::string_view text) const {
    std::vector<int> out;
    encode_into(text, /*with_specials=*/false, out);
    return out;
}

std::vector<int> ByteBpe::encode_chunked(std::string_view text, std::size_t chunk_bytes) const {
    std::vector<int> out;
    if (chunk_bytes == 0 || text.size() <= chunk_bytes) {
        encode_into(text, /*with_specials=*/true, out);
        return out;
    }
    std::vector<std::size_t> starts{0};
    for (std::size_t pos = chunk_bytes; pos < text.size();) {
        const std::size_t cut = find_cut(text, pos, specials_);
        if (cut >= text.size()) break;
        starts.push_back(cut);
        pos = cut + chunk_bytes;
    }
    std::vector<std::vector<int>> parts(starts.size());
    parallel_for(parts.size(), 1, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; i++) {
            const std::size_t stop = i + 1 < starts.size() ? starts[i + 1] : text.size();
            encode_into(text.substr(starts[i], stop - starts[i]), /*with_specials=*/true, parts[i]);
        }
    });
    std::size_t total = 0;
    for (const auto& part : parts) total += part.size();
    out.reserve(total);
    for (auto& part : parts) {
        out.insert(out.end(), part.begin(), part.end());
        std::vector<int>().swap(part);  // hand the memory back as we go
    }
    return out;
}

std::vector<int> ByteBpe::encode(std::string_view text) const {
    const auto threads = static_cast<std::size_t>(ThreadPool::instance().threads());
    if (text.size() < kParallelThreshold || threads <= 1) return encode_chunked(text, 0);
    const std::size_t chunk = std::clamp(text.size() / (threads * 8), kMinChunk, kMaxChunk);
    return encode_chunked(text, chunk);
}

std::string ByteBpe::decode(std::span<const int> ids) const {
    std::string out;
    for (const int id : ids) out += token_bytes(id);
    return out;
}

void ByteBpe::save(const std::string& path) const {
    std::string buf(kFileMagic);
    byte_io::put_u32(buf, kFileVersion);
    byte_io::put_u32(buf, pretok::kGpt2Unicode16);
    byte_io::put_u32(buf, static_cast<std::uint32_t>(vocab_size()));
    byte_io::put_u32(buf, static_cast<std::uint32_t>(merges_.size()));
    byte_io::put_u32(buf, static_cast<std::uint32_t>(specials_.size()));
    for (std::size_t k = 0; k < specials_.size(); k++) {
        byte_io::put_u32(buf, static_cast<std::uint32_t>(kByteTokens + merges_.size() + k));
        byte_io::put_u32(buf, static_cast<std::uint32_t>(specials_[k].size()));
        buf += specials_[k];
    }
    for (const auto& [left, right] : merges_) {
        byte_io::put_u32(buf, static_cast<std::uint32_t>(left));
        byte_io::put_u32(buf, static_cast<std::uint32_t>(right));
    }
    byte_io::put_u64(buf, fingerprint_);

    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open file for saving: " + path);
    file.write(buf.data(), static_cast<std::streamsize>(buf.size()));
    file.close();
    if (!file) throw std::runtime_error("Failed while writing tokenizer file: " + path);
}

ByteBpe ByteBpe::load(const std::string& path) {
    const std::string data = read_file(path);
    byte_io::Reader in(data, path);

    if (in.bytes(4, "magic") != kFileMagic) in.fail("not a GTOK tokenizer file");
    const std::uint32_t version = in.u32("version");
    if (version != kFileVersion) in.fail("unsupported version " + std::to_string(version));
    const std::uint32_t pretokenizer = in.u32("pre-tokenizer id");
    if (pretokenizer != pretok::kGpt2Unicode16) {
        in.fail("unknown pre-tokenizer id " + std::to_string(pretokenizer));
    }
    const std::uint32_t vocab = in.u32("vocab size");
    const std::uint32_t merge_count = in.u32("merge count");
    const std::uint32_t special_count = in.u32("special token count");
    if (vocab > kMaxVocab || merge_count > kMaxVocab || special_count > kMaxSpecials
        || std::uint64_t{kByteTokens} + merge_count + special_count != vocab) {
        in.fail("implausible header: vocab " + std::to_string(vocab) + ", "
                + std::to_string(merge_count) + " merges, " + std::to_string(special_count)
                + " special tokens");
    }

    std::vector<std::string> specials;
    specials.reserve(special_count);
    for (std::uint32_t k = 0; k < special_count; k++) {
        const std::uint32_t id = in.u32("special token id");
        if (id != kByteTokens + merge_count + k) {
            in.fail("special token " + std::to_string(k) + " has id " + std::to_string(id)
                    + "; special ids follow the merges");
        }
        const std::uint32_t len = in.u32("special token length");
        if (len == 0 || len > kMaxSpecialBytes) {
            in.fail("special token " + std::to_string(k) + " has an implausible length ("
                    + std::to_string(len) + " bytes)");
        }
        specials.emplace_back(in.bytes(len, "special token"));
    }

    // Each merge is 8 bytes; check they are there before reserving.
    if (in.remaining() / 8 < merge_count) in.fail("truncated reading merges");
    std::vector<Merge> merges;
    merges.reserve(merge_count);
    for (std::uint32_t r = 0; r < merge_count; r++) {
        const std::uint32_t left = in.u32("merge");
        const std::uint32_t right = in.u32("merge");
        if (left >= kByteTokens + r || right >= kByteTokens + r) {
            in.fail("merge " + std::to_string(r) + " joins tokens " + std::to_string(left) + " and "
                    + std::to_string(right) + ", which it does not follow");
        }
        merges.emplace_back(static_cast<int>(left), static_cast<int>(right));
    }
    const std::uint64_t stored = in.u64("fingerprint");
    if (!in.at_end()) in.fail("unexpected bytes after the fingerprint");

    ByteBpe tokenizer = [&] {
        try {
            return ByteBpe(std::move(merges), std::move(specials));
        } catch (const std::invalid_argument& e) {
            in.fail(e.what());
        }
    }();
    if (tokenizer.fingerprint() != stored) {
        in.fail("fingerprint does not match the contents (corrupt file)");
    }
    return tokenizer;
}

}  // namespace grad
