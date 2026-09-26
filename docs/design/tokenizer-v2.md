# Tokenizer v2: byte-level BPE

Status: implemented on `feature/tokenizer-v2`. v1 stays the tokenizer of every
existing checkpoint and keeps working byte for byte.

## Why

Tokenizer v1 (`BPETokenizer`) splits on whitespace with `iss >> word`, marks a
word start with `_`, and decodes `_` back to a space. Newlines, tabs and runs of
spaces never reach the model, and a literal underscore comes back as a space.
Merge selection breaks ties in `unordered_map` iteration order, so the same
corpus can give a different vocabulary under libc++ and libstdc++; training
rescans every word per merge, and encoding applies all ~16k merges in order to
every new word. The 70M TinyStories checkpoint and its token files depend on
v1, so v2 is an addition beside it, not a replacement.

## Design

**Vocabulary.** Ids 0-255 are the 256 bytes, ids 256.. are merges in the order
they were learned (merge `r` produces id `256 + r`), and the special tokens
follow the merges. Every byte string is encodable and `decode(encode(s)) == s`
for all inputs, including invalid UTF-8. `<|endoftext|>` is the one default
special; `add_special_token` appends more.

**Pre-tokenization** is a hand-written matcher for GPT-2's pattern
`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
(no `std::regex`), applied to the text between special tokens. Merges never
cross a pre-token boundary. Unicode is handled exactly rather than
approximated: `\p{L}` and `\p{N}` come from range tables generated from the
Unicode 16.0 database (`tools/gen_unicode_tables.py`: 677 letter and 144
number ranges, binary searched), and `\s` is the Unicode White_Space property
(25 code points). UTF-8 is decoded strictly (RFC 3629: no overlongs, surrogates or values above
U+10FFFF). The only deliberate difference from GPT-2 is the one GPT-2 cannot
have, since it works on decoded strings: each byte of an invalid sequence is its
own code point of class "other", i.e. it behaves like U+FFFD. The matcher was
checked against the Python `regex` module running the same pattern. It
matched on all 5.05M pieces of a 20MB TinyStories slice and on six random
Unicode texts of about 165k pieces each. The only differences were code points
unassigned in Unicode 16, which the module (on Unicode 17) treats as letters.
The unit test carries golden cases from that run.

**Special tokens** are matched before pre-tokenization (earliest occurrence,
longest on a tie), become single ids, are excluded from merge learning, and can
never be produced by byte merges, which only ever create merge ids.
`encode_ordinary` treats them as plain text.

**Training** counts unique pre-tokens with their frequencies, then runs
incremental BPE: pair counts live in a hash map, an index maps each pair to the
words containing it, and a max-heap with lazy deletion picks the next merge. A
merge rewrites only the words in its index entry and applies their net pair
deltas, so its cost is proportional to the words it touches, not the corpus.
Ties break deterministically: highest count, then the lexicographically smallest
(left bytes, right bytes), compared as unsigned bytes, then the smaller ids.
The last step is needed because two tokens can spell the same bytes (`ab`+`c`
and `a`+`bc`), and without it the order would not be total. Nothing depends on
hash iteration order, so a corpus gives the same vocabulary on every platform and
standard library. As in v1, corpora over 32MB learn merges from their first
32MB (cut at the last newline), whichever command trains the tokenizer.

**Encoding** is rank-based: within a pre-token, repeatedly merge the adjacent
pair of lowest rank (leftmost on a tie). Pre-tokens up to 16 bytes use a direct
scan; longer ones use a linked list plus a min-heap, O(n log n). Each chunk
keeps a pre-token cache. Inputs over 1MB are cut into chunks encoded in parallel
on the thread pool and concatenated in order. A cut is placed only right after a
`\n` that is a whitespace run on its own (the byte before it is ASCII and not
whitespace) and whose next code point is not whitespace, and never inside a
special token's occurrence. There the pattern makes the `\n` a token whether or
not text follows, and starts afresh after it, so the output is identical to a
serial encode for any chunking. The first version allowed any `\n` before
non-whitespace. That broke on `" \n"`, which is one token at the end of a chunk
but two (`" "`, `"\n"`) when text follows. A 1MB-chunk encode of the slice
caught it. The unit test now checks chunk sizes down to 1 byte.

**File format** (`<corpus>.bytebpe_<V>.tok`), all fields little-endian:

    "GTOK" | u32 version=2 | u32 pretokenizer=1 | u32 vocab_size
    u32 merge_count | u32 special_count
    special_count x (u32 id | u32 byte_length | bytes)
    merge_count  x (u32 left_id | u32 right_id)
    u64 fingerprint

Fields are written and read with explicit shifts, so the file is little-endian
on every host and no `std::endian` branch is needed. A read is validated like
the v1 cache: bounded counts and lengths, every read checked, merges may only
refer to bytes or earlier merges, special ids must follow the merges, the stored
fingerprint must match the recomputed one, and no trailing bytes are allowed. A
v1 cache starts with a u64 entry count below 2^24, so it can never begin with
`GTOK`. `load_tokenizer(path)` picks the format from the first four bytes.

**API.** `grad::Tokenizer` is a small interface: `encode`, `decode`,
`vocab_size`, `eos_id`, `special_id`, `fingerprint`, `save`. `BpeV1` wraps an
unchanged `BPETokenizer`, and `ByteBpe` is v2. The CLI and `TextGen` use only
the interface.

**Fingerprint.** A 64-bit FNV-1a hash over a canonical serialization of what
defines the tokenizer: a kind tag, then for v1 the id-ordered token table and
the merge list, and for v2 the pre-tokenizer id, the specials and the merge
list. It is the same on every platform and is frozen: changing it would orphan
the fingerprints in existing checkpoints.

**Checkpoints.** The fingerprint and kind go in an optional tagged trailer after
the last tensor (`"TKFP"`, u32 length, u32 kind, u64 hash), and the header stays
at format version 2. A checkpoint format v3 was the other option. The trailer
is better here because the existing loader stops after the final tensor and
never checks for EOF. A checkpoint written by this branch therefore still loads
in `main`'s binary, including the one that runs the v1 TinyStories jobs, and
resume pairs can move between the two binaries in both directions. Old checkpoints have no trailer and load
with the fingerprint unknown. Unknown trailer tags are skipped, and a truncated
trailer is an error. `train` records the fingerprint. `generate`, `chat`,
`eval`, `train ... resume` and warm starts refuse a mismatch, and warn when the
fingerprint is unknown.

**CLI.** `grad prepare <corpus> [vocab] [--tokenizer v2|v1]` defaults to v2.
v2 writes `<corpus>.bytebpe_<V>.tok` and `<corpus>.v2.<V>.{train,val}.bin`, so
v1's `<corpus>.tokenizer_<V>.cache` and `<corpus>.<V>.*.bin` are never touched.
The token files keep the uint16 `TOK1` format. Other commands choose the
tokenizer in this order: the kind recorded in the checkpoint (a `--tokenizer`
that contradicts it is an error); `--tokenizer`; v1 for a checkpoint without a
fingerprint, since only pre-v2 binaries wrote those; otherwise whichever kind's
files exist for the corpus and vocab. If both exist, the command stops with an
error that asks for `--tokenizer`. If neither exists, a new `train` uses v2. With v2, held-out prompts for corpora other than
Shakespeare start after the next `<|endoftext|>`, so TinyStories samples open a
story. v1 prompts are unchanged.

## Compatibility guarantees

- v1 caches, token files and checkpoints load and behave exactly as before.
  `train-fast` on Shakespeare with an existing `tokenizer_500.cache`, greedy
  generation from the 70M checkpoint, and its eval (1.6929) are unchanged.
- In a clean checkout, `train-fast` now trains a v2 tokenizer, so its metrics
  differ from the v1 golden (v2 golden: metrics-column md5 `0677d2a0...`).
  `--tokenizer v1` restores the old path.
- A checkpoint written with the trailer loads in `main`'s binary and generates
  the same text there.
- Vocabularies must fit in 65536 ids, since token files store uint16.

## Measurements

On a 20MB TinyStories slice at vocab 16000 (M3 Pro, `nice -n 19`, 4 threads,
machine in use):

| | v1 | v2 |
|---|---|---|
| train tokenizer | 99.6 s | 0.26-0.61 s |
| encode | 7.3 MB/s | 55-115 MB/s serial, 130-220 MB/s on 4 threads |
| tokens | 4.15M (5.0 bytes/token, whitespace dropped) | 5.01M (4.19 bytes/token, lossless) |

For the full 1.9GB corpus, v2 learns merges from 32MB in about 1 s, and encoding
at 130-220 MB/s takes 9-15 s, against about 4.5 minutes for v1. Peak memory is
about the text plus one copy of the ids (1.9GB + 1.8GB), because each chunk's
result is freed as it is concatenated. v2 needs about 20% more tokens for
the same text. It keeps every newline, and GPT-2's pattern splits punctuation
from words, where v1 learns tokens like `_girl.`. A 328M-token budget therefore
covers about 17% less text.
