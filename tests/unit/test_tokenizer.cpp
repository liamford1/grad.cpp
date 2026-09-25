// Tokenizer tests: the GPT-2 pre-tokenizer against golden splits from the
// reference regex, ByteBpe round trips, training determinism and its
// tie-break rule, rank-based encoding against the merge-in-order
// definition, chunked encoding, the GTOK file format, and v1 behavior.
//
//   test_tokenizer                 everything that needs no external file
//   test_tokenizer v1-cache PATH   v1 against ids recorded from main's
//                                  binary with the TinyStories 16000 cache;
//                                  exits 77 (skipped) if PATH is missing

#include "grad/tokenizer/bpe_tokenizer.h"
#include "grad/tokenizer/bpe_v1.h"
#include "grad/tokenizer/byte_bpe.h"
#include "grad/tokenizer/pretokenize.h"
#include "grad/tokenizer/tokenizer.h"

#include "../test_util.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using grad::BpeV1;
using grad::ByteBpe;
namespace pretok = grad::pretok;

namespace {

// splitmix64: a fully specified generator, so the synthetic corpora (and
// the fingerprint of the tokenizer trained on one) are the same on every
// platform, unlike std::uniform_int_distribution.
class Rng {
public:
    explicit Rng(std::uint64_t seed) : state_(seed) {}
    std::uint64_t next() {
        std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    std::size_t below(std::size_t n) { return static_cast<std::size_t>(next() % n); }

private:
    std::uint64_t state_;
};

const std::vector<std::string>& word_pool() {
    static const std::vector<std::string> pool = {"Once",
                                                  " upon",
                                                  " a",
                                                  " time",
                                                  ",",
                                                  " there",
                                                  " was",
                                                  " little",
                                                  " girl",
                                                  " named",
                                                  " Lily",
                                                  ".",
                                                  "\n",
                                                  " She",
                                                  " loved",
                                                  " to",
                                                  " play",
                                                  " outside",
                                                  " with",
                                                  " her",
                                                  " dog",
                                                  "!",
                                                  " \"",
                                                  "Hi",
                                                  "\"",
                                                  " said",
                                                  " Tom",
                                                  "'s",
                                                  " don't",
                                                  "  ",
                                                  "\t",
                                                  "\r\n",
                                                  " foo_bar",
                                                  " __init__",
                                                  " 123",
                                                  " 4.5",
                                                  " caf\xc3\xa9",
                                                  " \xf0\x9f\x98\x80",
                                                  " \xe6\x97\xa5\xe6\x9c\xac",
                                                  "\n\n",
                                                  "   ",
                                                  " the",
                                                  " end",
                                                  "<|endoftext|>",
                                                  "\n"};
    return pool;
}

// Pool pieces mixed with made-up words of 1-4 syllables, so there are
// enough distinct pairs to learn several hundred merges from.
std::string synthetic_corpus(std::uint64_t seed, std::size_t words) {
    static const std::vector<std::string> syllables = {
        "ka", "to", "ri", "mo", "an", "el", "su", "ph", "qu", "ing", "ed", "th", "st", "ou", "Ã©"};
    Rng rng(seed);
    std::string text;
    for (std::size_t i = 0; i < words; i++) {
        if (rng.below(2) == 0) {
            text += word_pool()[rng.below(word_pool().size())];
            continue;
        }
        text += ' ';
        const std::size_t n = 1 + rng.below(4);
        for (std::size_t k = 0; k < n; k++) text += syllables[rng.below(syllables.size())];
    }
    return text;
}

std::string random_bytes(Rng& rng, std::size_t n) {
    std::string s(n, '\0');
    for (char& c : s) c = static_cast<char>(rng.next() & 0xFF);
    return s;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void write_file(const std::string& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

template <typename Fn>
bool throws(Fn&& fn) {
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

// The definition encode must agree with: per pre-token, apply every merge
// in training order, each left to right.
std::vector<int> reference_encode(const ByteBpe& tok, std::string_view text) {
    std::vector<int> out;
    for (const std::string_view piece : pretok::split(text)) {
        std::vector<int> s;
        for (const char c : piece) s.push_back(static_cast<unsigned char>(c));
        for (std::size_t r = 0; r < tok.merges().size() && s.size() > 1; r++) {
            const auto [left, right] = tok.merges()[r];
            std::vector<int> merged;
            for (std::size_t i = 0; i < s.size(); i++) {
                if (i + 1 < s.size() && s[i] == left && s[i + 1] == right) {
                    merged.push_back(ByteBpe::kByteTokens + static_cast<int>(r));
                    i++;
                } else {
                    merged.push_back(s[i]);
                }
            }
            s = std::move(merged);
        }
        out.insert(out.end(), s.begin(), s.end());
    }
    return out;
}

// ---- Pre-tokenizer -----------------------------------------------------

struct SplitCase {
    std::string_view text;
    std::vector<std::string_view> pieces;
};

// Generated with the reference: Python's `regex` module and GPT-2's
// pattern. The differential run behind the design doc also compared the
// full TinyStories slice and random Unicode text and found no difference
// on any code point assigned in Unicode 16.
const std::vector<SplitCase>& golden_splits() {
    static const std::vector<SplitCase> cases = {
        {"Hello world", {"Hello", " world"}},
        {"Hello  world", {"Hello", " ", " world"}},
        {"Hello   world", {"Hello", "  ", " world"}},
        {"a\nb", {"a", "\n", "b"}},
        {"a\n\nb", {"a", "\n", "\n", "b"}},
        {"a \nb", {"a", " ", "\n", "b"}},
        {"a\n\n", {"a", "\n\n"}},
        {"  leading", {" ", " leading"}},
        {"trailing  ", {"trailing", "  "}},
        {"tab\tsep", {"tab", "\t", "sep"}},
        {"\t\tx", {"\t", "\t", "x"}},
        {"CRLF\r\nline", {"CRLF", "\r", "\n", "line"}},
        {"don't", {"don", "'t"}},
        {"I'm here", {"I", "'m", " here"}},
        {"they're", {"they", "'re"}},
        {"we've", {"we", "'ve"}},
        {"you'll", {"you", "'ll"}},
        {"he'd", {"he", "'d"}},
        {"it's", {"it", "'s"}},
        {"IT'S", {"IT", "'", "S"}},
        {"'hello'", {"'", "hello", "'"}},
        {"!'s", {"!'", "s"}},
        {"x 's", {"x", " '", "s"}},
        {"rock'n'roll", {"rock", "'", "n", "'", "roll"}},
        {"foo_bar baz", {"foo", "_", "bar", " baz"}},
        {"__init__", {"__", "init", "__"}},
        {"a_b", {"a", "_", "b"}},
        {"123 4567 89", {"123", " 4567", " 89"}},
        {"abc123def", {"abc", "123", "def"}},
        {"3.14", {"3", ".", "14"}},
        {" $5.00!", {" $", "5", ".", "00", "!"}},
        {"...", {"..."}},
        {"?!", {"?!"}},
        {"\"Hi,\" she said.", {"\"", "Hi", ",\"", " she", " said", "."}},
        {"(a) [b] {c}", {"(", "a", ")", " [", "b", "]", " {", "c", "}"}},
        {"e-mail", {"e", "-", "mail"}},
        {"--flag=value", {"--", "flag", "=", "value"}},
        {"<|endoftext|>", {"<|", "endoftext", "|>"}},
        {"x\x0b\x0cy", {"x", "\x0b", "\x0c", "y"}},
        {"\x1c\x1dz", {"\x1c\x1d", "z"}},
        {" ", {" "}},
        {"", {}},
        {"caf\xc3\xa9", {"caf\xc3\xa9"}},
        {"e\xcc\x81", {"e", "\xcc\x81"}},
        {"\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e\xe3\x81\xa7\xe3\x81\x99",
         {"\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e\xe3\x81\xa7\xe3\x81\x99"}},
        {"na\xc3\xafve r\xc3\xa9sum\xc3\xa9", {"na\xc3\xafve", " r\xc3\xa9sum\xc3\xa9"}},
        {"\xc2\xa0nbsp", {"\xc2\xa0", "nbsp"}},
        {"a\xe3\x80\x80\x62", {"a", "\xe3\x80\x80", "b"}},
        {"\xf0\x9f\x98\x80 smile", {"\xf0\x9f\x98\x80", " smile"}},
        {"x\xc2\xb2 + \xc2\xbd", {"x", "\xc2\xb2", " +", " \xc2\xbd"}},
        {"\xe2\x85\xa7 \xd9\xa3", {"\xe2\x85\xa7", " \xd9\xa3"}},
        {"\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7", {"\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7"}},
    };
    return cases;
}

void test_pretokenizer_golden() {
    for (const SplitCase& c : golden_splits()) {
        const std::vector<std::string_view> got = pretok::split(c.text);
        if (!CHECK(got == c.pieces)) {
            std::cerr << "  splitting \"" << c.text << "\" gave " << got.size() << " pieces"
                      << std::endl;
        }
    }
}

void test_character_classes() {
    using pretok::CharClass;
    CHECK(pretok::classify('a') == CharClass::Letter);
    CHECK(pretok::classify('Z') == CharClass::Letter);
    CHECK(pretok::classify('7') == CharClass::Number);
    CHECK(pretok::classify('_') == CharClass::Other);
    CHECK(pretok::classify('\'') == CharClass::Other);
    for (const std::uint32_t ws : {0x09u, 0x0Au, 0x0Bu, 0x0Cu, 0x0Du, 0x20u, 0x85u, 0xA0u, 0x1680u,
                                   0x2000u, 0x200Au, 0x2028u, 0x2029u, 0x202Fu, 0x205Fu, 0x3000u}) {
        CHECK(pretok::classify(ws) == CharClass::Space);
    }
    CHECK(pretok::classify(0x1C) == CharClass::Other);    // not White_Space
    CHECK(pretok::classify(0x200B) == CharClass::Other);  // zero-width space is Cf
    CHECK(pretok::classify(0xE9) == CharClass::Letter);   // e acute
    CHECK(pretok::classify(0x0301) == CharClass::Other);  // combining acute (Mn)
    CHECK(pretok::classify(0x4E2D) == CharClass::Letter);
    CHECK(pretok::classify(0xB2) == CharClass::Number);    // superscript two (No)
    CHECK(pretok::classify(0x2167) == CharClass::Number);  // Roman numeral eight (Nl)
    CHECK(pretok::classify(0x0663) == CharClass::Number);  // Arabic-Indic three (Nd)
    CHECK(pretok::classify(0x1F600) == CharClass::Other);  // emoji (So)
    CHECK(pretok::classify(0x10FFFF) == CharClass::Other);
}

void test_utf8_decoding() {
    using pretok::CharClass;
    const auto one_invalid_byte = [](std::string_view s) {
        const pretok::Codepoint cp = pretok::decode_at(s, 0);
        return cp.length == 1 && cp.cls == CharClass::Other;
    };
    CHECK(one_invalid_byte("\xC0\xAF"));          // overlong '/'
    CHECK(one_invalid_byte("\xE0\x80\xAF"));      // overlong
    CHECK(one_invalid_byte("\xED\xA0\x80"));      // surrogate U+D800
    CHECK(one_invalid_byte("\xF4\x90\x80\x80"));  // U+110000
    CHECK(one_invalid_byte("\xE6\x97"));          // truncated
    CHECK(one_invalid_byte("\x80"));              // stray continuation
    CHECK(one_invalid_byte("\xFF"));
    const pretok::Codepoint e = pretok::decode_at("\xc3\xa9", 0);
    CHECK(e.value == 0xE9 && e.length == 2 && e.cls == CharClass::Letter);
    const pretok::Codepoint top = pretok::decode_at("\xF4\x8F\xBF\xBF", 0);
    CHECK(top.value == 0x10FFFF && top.length == 4);
}

void test_pretokenizer_partitions() {
    Rng rng(7);
    for (int trial = 0; trial < 300; trial++) {
        std::string text = random_bytes(rng, rng.below(200));
        if (trial % 2 == 0) text += synthetic_corpus(rng.next(), 20);
        std::string joined;
        bool nonempty = true;
        for (const std::string_view piece : pretok::split(text)) {
            nonempty = nonempty && !piece.empty();
            joined += piece;
        }
        CHECK(nonempty);
        CHECK(joined == text);
    }
}

// ---- ByteBpe -----------------------------------------------------------

const ByteBpe& trained() {
    static const ByteBpe tok = ByteBpe::train(synthetic_corpus(1, 20000), 700);
    return tok;
}

void test_byte_bpe_round_trips() {
    const ByteBpe& tok = trained();
    CHECK(tok.vocab_size() == 700);
    const int eos = ByteBpe::kByteTokens + static_cast<int>(tok.merges().size());
    CHECK(tok.eos_id() == eos);
    CHECK(tok.special_id("<|endoftext|>") == eos);
    CHECK(!tok.special_id("<eos>").has_value());

    const std::vector<std::string> cases = {
        "",
        "a",
        "Hello world",
        "   leading and trailing   ",
        "tabs\tand\t\ttabs",
        "lines\nand\n\n\nblank lines\n",
        "CRLF\r\nline\r\n\r\n",
        "literal_underscores __init__ _a_ ___",
        "multi   spaces  \t \n mixed",
        "caf\xc3\xa9 na\xc3\xafve \xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e",
        "emoji \xf0\x9f\x98\x80\xf0\x9f\x91\x8d\xf0\x9f\x8f\xbd!",
        "invalid \xff\xfe \xc0\xaf \xed\xa0\x80 \xe6\x97 \x80\x80",
        std::string("nul\0byte", 8),
        "<|endoftext|>",
        "a<|endoftext|>b",
        "<|endoftext|><|endoftext|><|endoftext|>",
        "end.\n<|endoftext|>\nOnce upon",
        "<|endoftext",
        "<|endoftext|",
        "<<|endoftext|>>",
        std::string(1000, ' '),
        std::string(300, 'a') + std::string(300, '\n'),
    };
    for (const std::string& s : cases) {
        const std::vector<int> ids = tok.encode(s);
        CHECK(tok.decode(ids) == s);
        for (const int id : ids) CHECK(id >= 0 && id < tok.vocab_size());
        CHECK(tok.decode(tok.encode_ordinary(s)) == s);
    }

    Rng rng(11);
    for (int trial = 0; trial < 500; trial++) {
        std::string s = random_bytes(rng, rng.below(300));
        if (trial % 3 == 0) s += "<|endoftext|>" + random_bytes(rng, 5);
        CHECK(tok.decode(tok.encode(s)) == s);
    }
}

void test_special_tokens() {
    const ByteBpe& tok = trained();
    const int eos = tok.eos_id().value_or(-1);
    CHECK(eos >= 0);
    CHECK(tok.encode("<|endoftext|>") == std::vector<int>{eos});
    CHECK(tok.encode("<|endoftext|><|endoftext|>") == (std::vector<int>{eos, eos}));

    const std::vector<int> adjacent = tok.encode("x<|endoftext|>y");
    CHECK(adjacent.size() == 3 && adjacent[1] == eos);

    // Ordinary encoding spells it in bytes, and no merge ever yields a
    // special id.
    const std::vector<int> ordinary = tok.encode_ordinary("<|endoftext|>");
    CHECK(ordinary.size() > 1);
    for (const int id : ordinary) CHECK(id < eos);
    for (const auto& [left, right] : tok.merges()) CHECK(left < eos && right < eos);

    ByteBpe more = tok;
    const std::uint64_t before = more.fingerprint();
    const int sep = more.add_special_token("<|sep|>");
    CHECK(sep == tok.vocab_size());
    CHECK(more.vocab_size() == tok.vocab_size() + 1);
    CHECK(more.fingerprint() != before);
    CHECK(more.encode("a<|sep|>b<|endoftext|>") == (std::vector<int>{'a', sep, 'b', eos}));
    CHECK(throws([&] { more.add_special_token("<|sep|>"); }));
    CHECK(throws([&] { more.add_special_token(""); }));
    CHECK(more.vocab_size() == tok.vocab_size() + 1);  // failed adds change nothing

    // Overlapping specials: the earliest occurrence wins, then the longest.
    ByteBpe overlap(std::vector<std::string>{"<a>", "<a><b>", "b>"});
    const int a = overlap.special_id("<a>").value_or(-1);
    const int ab = overlap.special_id("<a><b>").value_or(-1);
    const int b = overlap.special_id("b>").value_or(-1);
    CHECK(a >= 0 && ab >= 0 && b >= 0);
    CHECK(overlap.encode("<a><b>") == std::vector<int>{ab});
    CHECK(overlap.encode("<a>b>") == (std::vector<int>{a, b}));
}

void test_encode_matches_merge_order() {
    const ByteBpe& tok = trained();
    Rng rng(5);
    for (int trial = 0; trial < 200; trial++) {
        std::string s = synthetic_corpus(rng.next(), 30);
        // Long pieces take the heap path: letter runs, space runs, and
        // long runs of "other" bytes.
        if (trial % 4 == 0) s += " " + std::string(40 + rng.below(40), 'a');
        if (trial % 4 == 1) s += std::string(30 + rng.below(30), ' ') + "x";
        if (trial % 4 == 2) s += random_bytes(rng, 64);
        if (trial % 4 == 3) s += " supercalifragilisticexpialidocious_is_a_long_word";
        CHECK(tok.encode_ordinary(s) == reference_encode(tok, s));
    }
    // Every training word was merged by the same rule.
    CHECK(tok.encode_ordinary(" little") == reference_encode(tok, " little"));
}

void test_chunked_encoding_is_deterministic() {
    const ByteBpe& tok = trained();
    Rng rng(9);
    std::string text = synthetic_corpus(3, 6000);
    text += random_bytes(rng, 500) + "\n\n  \n" + "\xe3\x80\x80\n" + "x\n<|endoftext|>\nY";
    const std::vector<int> serial = tok.encode_chunked(text, 0);
    for (const std::size_t chunk :
         {std::size_t{1}, std::size_t{7}, std::size_t{64}, std::size_t{1000}}) {
        if (!CHECK(tok.encode_chunked(text, chunk) == serial)) {
            std::cerr << "  chunk size " << chunk << std::endl;
        }
    }
    CHECK(tok.encode(text) == serial);
    CHECK(tok.decode(serial) == text);
}

void test_training_determinism() {
    const std::string corpus = synthetic_corpus(2, 8000);
    const ByteBpe a = ByteBpe::train(corpus, 500);
    const ByteBpe b = ByteBpe::train(corpus, 500);
    CHECK(a.merges() == b.merges());
    CHECK(a.fingerprint() == b.fingerprint());

    // Merge learning sees a multiset of pre-tokens, so reordering lines
    // (each ends in "\n" and starts with a non-space) changes nothing.
    std::vector<std::string> lines;
    const std::string text =
        "the cat sat\nthe dog ran fast\na cat and a dog\nthe end of the story\nlittle girl\n"
        "she loved to play\nhe said hello\ntheir cat sat on the mat\n";
    for (std::size_t pos = 0; pos < text.size();) {
        const std::size_t nl = text.find('\n', pos);
        lines.push_back(text.substr(pos, nl + 1 - pos));
        pos = nl + 1;
    }
    std::string forward;
    std::string backward;
    for (std::size_t i = 0; i < lines.size(); i++) {
        for (int repeat = 0; repeat < static_cast<int>(i % 3) + 1; repeat++) {
            forward += lines[i];
            backward.insert(0, lines[i]);
        }
    }
    const ByteBpe f = ByteBpe::train(forward, 300);
    const ByteBpe r = ByteBpe::train(backward, 300);
    CHECK(f.merges().size() == 300 - 257);
    CHECK(f.merges() == r.merges());

    // Frozen: a change here means the same corpus now gives a different
    // vocabulary (or the fingerprint definition moved), on this platform or
    // relative to the one that recorded it.
    const ByteBpe golden = ByteBpe::train(synthetic_corpus(42, 5000), 400);
    CHECK(golden.vocab_size() == 400);
    CHECK(golden.fingerprint() == 0x539365143cd9caeeULL);
    CHECK(ByteBpe({{97, 98}}, {"<|endoftext|>"}).fingerprint() == 0x08b0b6b208dca2a9ULL);
}

void test_tie_break() {
    using Merges = std::vector<ByteBpe::Merge>;
    // Equal counts, equal left bytes: the smaller right bytes first.
    CHECK(ByteBpe::train("ab\nac", 258, {}).merges() == (Merges{{'a', 'b'}, {'a', 'c'}}));
    // A higher count beats any tie-break; then " " < "c" and "ab" < "c".
    CHECK(ByteBpe::train("ab ab cd", 260, {}).merges()
          == (Merges{{'a', 'b'}, {' ', 256}, {' ', 'c'}, {258, 'd'}}));
    // Bytes compare unsigned: 'z' (0x7A) < 0xA9 < 0xC3.
    CHECK(ByteBpe::train("zb\n\xc3\xa9\x62", 259, {}).merges()
          == (Merges{{'z', 'b'}, {0xA9, 'b'}, {0xC3, 257}}));
    // Too small a vocabulary for the bytes and specials is an error; one
    // with room to spare stops when the pairs run out.
    CHECK(throws([] { (void)ByteBpe::train("abc", 256); }));
    CHECK(ByteBpe::train("abc", 1000, {}).vocab_size() == 258);
}

// ---- File format -------------------------------------------------------

void test_save_load(const std::filesystem::path& dir) {
    const ByteBpe& tok = trained();
    const std::string path = (dir / "tok.tok").string();
    tok.save(path);
    CHECK(grad::detect_tokenizer_file(path) == grad::TokenizerKind::ByteBpe);
    const ByteBpe loaded = ByteBpe::load(path);
    CHECK(loaded.merges() == tok.merges());
    CHECK(loaded.specials() == tok.specials());
    CHECK(loaded.fingerprint() == tok.fingerprint());
    const std::string sample = synthetic_corpus(8, 500);
    CHECK(loaded.encode(sample) == tok.encode(sample));

    const std::unique_ptr<grad::Tokenizer> any = grad::load_tokenizer(path);
    CHECK(any->kind() == grad::TokenizerKind::ByteBpe);
    CHECK(any->identity() == tok.identity());
    CHECK(any->decode(any->encode(sample)) == sample);

    // Saving what was loaded reproduces the file byte for byte.
    const std::string again = (dir / "again.tok").string();
    loaded.save(again);
    CHECK(read_file(again) == read_file(path));
}

void test_corrupt_files(const std::filesystem::path& dir) {
    const ByteBpe tok = ByteBpe::train(synthetic_corpus(4, 2000), 300);
    const std::string path = (dir / "good.tok").string();
    tok.save(path);
    const std::string good = read_file(path);
    const std::string bad = (dir / "bad.tok").string();
    const auto rejected = [&](const std::string& bytes) {
        write_file(bad, bytes);
        return throws([&] { (void)ByteBpe::load(bad); });
    };
    const auto with_u32 = [&](std::size_t offset, std::uint32_t v) {
        std::string bytes = good;
        for (std::size_t k = 0; k < 4; k++)
            bytes[offset + k] = static_cast<char>((v >> (8 * k)) & 0xFFu);
        return bytes;
    };

    CHECK(!rejected(good));
    bool every_truncation = true;
    for (std::size_t len = 0; len < good.size(); len++) {
        every_truncation = every_truncation && rejected(good.substr(0, len));
    }
    CHECK(every_truncation);
    CHECK(rejected(good + "x"));
    CHECK(rejected("GTOX" + good.substr(4)));
    CHECK(rejected(with_u32(4, 3)));             // version
    CHECK(rejected(with_u32(8, 99)));            // pre-tokenizer id
    CHECK(rejected(with_u32(12, 301)));          // vocab disagrees with the counts
    CHECK(rejected(with_u32(16, 0xFFFFFFF0u)));  // huge merge count
    CHECK(rejected(with_u32(20, 5000)));         // too many specials
    CHECK(rejected(with_u32(24, 256)));          // special id not after the merges
    CHECK(rejected(with_u32(28, 0)));            // zero-length special
    CHECK(rejected(with_u32(28, 0x7FFFFFFFu)));  // huge special
    const std::size_t merges_at = 32 + tok.specials()[0].size();
    CHECK(rejected(with_u32(merges_at, 256)));     // merge 0 refers to itself
    CHECK(rejected(with_u32(merges_at + 8, 98)));  // changes a merge: fingerprint mismatch
    std::string flipped = good;
    flipped.back() = static_cast<char>(flipped.back() ^ 1);
    CHECK(rejected(flipped));
    CHECK(throws([&] { (void)ByteBpe::load((dir / "missing.tok").string()); }));

    CHECK(throws([] { (void)ByteBpe({{300, 1}}, {}); }));
    CHECK(throws([] { (void)ByteBpe({{97, 98}, {97, 98}}, {}); }));
    CHECK(throws([] { (void)ByteBpe(std::vector<std::string>{"x", "x"}); }));
    CHECK(throws([&] { (void)tok.decode(std::vector<int>{tok.vocab_size()}); }));
    CHECK(throws([&] { (void)tok.decode(std::vector<int>{-1}); }));
}

// ---- v1 ------------------------------------------------------------------

void test_v1_wrapper(const std::filesystem::path& dir) {
    grad::BPETokenizer raw(60);
    raw.train("the cat sat on the mat and the cat ran to the hat");
    const std::string path = (dir / "v1.cache").string();
    raw.save(path);

    const BpeV1 v1 = BpeV1::load(path);
    const std::string text = "the cat sat\n\ton  the_mat";
    CHECK(v1.encode(text) == raw.encode(text));
    CHECK(v1.decode(v1.encode(text)) == raw.decode(raw.encode(text)));
    CHECK(v1.vocab_size() == raw.getCurrentVocabSize());
    CHECK(v1.eos_id() == 1);
    CHECK(v1.special_id("<unk>") == 2);
    CHECK(!v1.special_id("<|endoftext|>").has_value());

    CHECK(grad::detect_tokenizer_file(path) == grad::TokenizerKind::BpeV1);
    const std::unique_ptr<grad::Tokenizer> any = grad::load_tokenizer(path);
    CHECK(any->kind() == grad::TokenizerKind::BpeV1);
    CHECK(any->fingerprint() == v1.fingerprint());
    CHECK(any->encode(text) == raw.encode(text));

    // The fingerprint depends on the tables, not on the cache's hash-map
    // order: re-saving the loaded tokenizer keeps it.
    const std::string resaved = (dir / "v1b.cache").string();
    v1.save(resaved);
    CHECK(BpeV1::load(resaved).fingerprint() == v1.fingerprint());
    grad::BPETokenizer other(60);
    other.train("a different corpus entirely");
    CHECK(BpeV1(std::move(other)).fingerprint() != v1.fingerprint());
}

struct V1Golden {
    std::string text;
    std::vector<int> ids;
    std::string decoded;
};

// Recorded from main (0377275) with the TinyStories 16000 cache, before
// any v2 change.
int test_v1_cache(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        std::cout << "v1 cache " << path << " not found; skipping" << std::endl;
        return 77;
    }
    const std::vector<V1Golden> golden = {
        {"Once upon a time", {67, 93, 212, 304, 109, 461}, "Once upon a time"},
        {"Once upon a time, there was a little girl named Lily.\nShe loved to play outside.",
         {67, 93, 212, 304, 109, 332, 260, 133, 109, 259, 312, 361, 507, 161, 368, 116, 216, 1164},
         "Once upon a time, there was a little girl named Lily. She loved to play outside."},
        {"Hello   world\tfoo_bar  baz\r\n",
         {60, 441, 94, 1177, 122, 181, 1654, 1192, 105},
         "Hello world foo bar baz"},
        {"<|endoftext|> The end.", {234, 232, 106, 51, 137, 1062}, "<|endoftext|> The end."},
        {"", {}, ""},
        {"Tom said, \"I'm happy!\" 123 ... caf\xc3\xa9 \xf0\x9f\x98\x80",
         {72, 134, 265, 1046, 10280, 7302, 40, 41, 79, 11223, 12260, 21, 14, 79, 2, 2, 7, 3},
         "Tom said, \"I'm happy!\" 123 ... caf\xc3\xa9 <unk><unk>\x98\x80"},
    };
    const std::unique_ptr<grad::Tokenizer> tok = grad::load_tokenizer(path);
    CHECK(tok->kind() == grad::TokenizerKind::BpeV1);
    CHECK(tok->vocab_size() == 16000);
    CHECK(tok->fingerprint() == 0xba36be291b13ad8eULL);
    grad::BPETokenizer raw(16000);
    raw.load(path);
    for (const V1Golden& g : golden) {
        CHECK(tok->encode(g.text) == g.ids);
        CHECK(raw.encode(g.text) == g.ids);
        CHECK(tok->decode(g.ids) == g.decoded);
    }
    return test_util::exit_code();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::string_view(argv[1]) == "v1-cache") return test_v1_cache(argv[2]);

    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "grad_test_tokenizer";
    std::filesystem::create_directories(dir);

    test_pretokenizer_golden();
    test_character_classes();
    test_utf8_decoding();
    test_pretokenizer_partitions();
    test_byte_bpe_round_trips();
    test_special_tokens();
    test_encode_matches_merge_order();
    test_chunked_encoding_is_deterministic();
    test_training_determinism();
    test_tie_break();
    test_save_load(dir);
    test_corrupt_files(dir);
    test_v1_wrapper(dir);

    std::filesystem::remove_all(dir);
    return test_util::exit_code();
}
