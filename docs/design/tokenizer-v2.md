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
Unicode 16.0 database (`tools/gen_unicode_tables.py`, about 900 ranges, binary
searched), and `\s` is the Unicode White_Space property (25 code points). UTF-8
is decoded strictly (RFC 3629: no overlongs, surrogates or values above
U+10FFFF). The only deliberate difference from GPT-2 is the one GPT-2 cannot
have, since it works on decoded strings: each byte of an invalid sequence is its
own code point of class "other", i.e. it behaves like U+FFFD. The matcher was
checked against the Python `regex` module on the TinyStories slice and on random
Unicode and byte strings; the unit test carries golden cases from that run.

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
(left bytes, right bytes), compared as unsigned bytes. Nothing depends on hash
iteration order, so a corpus gives the same vocabulary on every platform and
standard library. As in v1, corpora over 32MB learn merges from their first
32MB (cut at the last newline), whichever command trains the tokenizer.

**Encoding** is rank-based: within a pre-token, repeatedly merge the adjacent
pair of lowest rank (leftmost on a tie). Pre-tokens up to 16 bytes use a direct
scan; longer ones use a linked list plus a min-heap, O(n log n). Each chunk
keeps a pre-token cache. Inputs over 1MB are cut into chunks encoded in parallel
on the thread pool and concatenated in order. A cut is placed only right after a
`\n` whose next code point is not whitespace, and never inside a special
token's occurrence. The regex can never join tokens across such a point: a
newline is only ever part of a whitespace token, and that token ends before the
following non-whitespace. So the output is identical to a serial encode for any
chunking. The unit test checks this with 64-byte chunks.

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
at format version 2. The task suggested a checkpoint v3. The trailer is better
here because the existing loader stops after the final tensor and never checks
for EOF. A checkpoint written by this branch therefore still loads in `main`'s
binary, including the binary running tonight's v1 run, and resume pairs can move
between the two in both directions. Old checkpoints have no trailer and load
with the fingerprint unknown. Unknown trailer tags are skipped, and a truncated
trailer is an error. `train` records the fingerprint. `generate`, `chat`,
`eval`, `train ... resume` and warm starts refuse a mismatch, and warn when the
fingerprint is unknown.

**CLI.** `grad prepare <corpus> [vocab] [--tokenizer v2|v1]` defaults to v2.
v2 writes `<corpus>.bytebpe_<V>.tok` and `<corpus>.v2.<V>.{train,val}.bin`, so
v1's `<corpus>.tokenizer_<V>.cache` and `<corpus>.<V>.*.bin` are never touched.
The token files keep the uint16 `TOK1` format. Other commands choose the
tokenizer in this order: `--tokenizer`; the kind recorded in the checkpoint; for
a checkpoint without a fingerprint, v1, since only pre-v2 binaries wrote those;
otherwise whichever kind's files exist for the corpus and vocab. If both exist,
the command stops with an error that asks for `--tokenizer`. If neither exists,
a new `train` uses v2. An explicit `--tokenizer` that contradicts the
checkpoint is an error.

## Compatibility guarantees

- v1 caches, token files and checkpoints load and behave exactly as before.
  `train-fast` on Shakespeare with an existing `tokenizer_500.cache`, greedy
  generation from the 70M checkpoint, and its eval (1.6929) are unchanged.
- In a clean checkout, `train-fast` now trains a v2 tokenizer, so its metrics
  differ from the v1 golden. `--tokenizer v1` restores the old path.
- Vocabularies must fit in 65536 ids, since token files store uint16.
