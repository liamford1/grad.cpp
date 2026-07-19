# transformer-from-scratch

[![CI](https://github.com/liamford1/transformer-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/liamford1/transformer-from-scratch/actions/workflows/ci.yml)

A GPT-style language model implemented from scratch in C++17 — tensors, reverse-mode autograd, multi-head attention, Adam, and a BPE tokenizer, with no ML frameworks. The only external dependency is a BLAS library (Apple Accelerate on macOS, OpenBLAS on Linux) for fast matrix multiplication.

A ~22M-parameter model trained with this code on the Tiny Shakespeare corpus produces text like this (sampled at temperature 0.8; line breaks added at speaker changes for readability, text otherwise unedited):

> **JULIET:** me, will find in my heart my friends, And I say it is be a rap o' not the gravly up the sacance show it be lot, But thou wastenes.
> **ISABELLA:** Not him all women With common.
> **CORIOLANUS:** Thou didst: Then say forth to helps; but I know and honour of's lave you; whether sir!
> **LUCIO:** She that hath made buriedion.
> **QUEEN ELIZABETH:** Falive a cup in A matchly wash'd by his soul!

Not going to win a Pulitzer, but every gradient that trained it was derived and implemented by hand.

## What's inside

- **Tensor library** (`tensor.h/cpp`) — 2D/3D float tensors with BLAS-backed matmul, broadcasting, and numerically stable softmax/log-softmax
- **Reverse-mode autograd** (`variable.h/cpp`) — dynamic computation graph with hand-derived backward passes for every op, validated against numerical gradients
- **Transformer components** — multi-head self-attention with causal masking, pre-LayerNorm residual blocks, GELU feed-forward, learned positional embeddings, and weight tying between the token embedding and the output projection
- **BPE tokenizer** — byte-pair encoding trained on the corpus, with caching so repeat runs start instantly
- **Training stack** — AdamW (decoupled weight decay, matrices only) with linear warmup + cosine LR decay, gradient clipping, dropout, a shuffling DataLoader, checkpoint save/load, live loss/grad-norm metrics, and held-out validation perplexity every 250 steps
- **Text generation** — KV-cached incremental decoding (`inference.h`), greedy or sampled with temperature, top-k, top-p, and a repetition penalty

About 5,400 lines of implementation and 1,200 lines of tests.

## Model architecture

The trained checkpoint uses a standard GPT-style decoder-only configuration:

| | |
|---|---|
| Parameters | ~22M |
| Layers | 6 (pre-LN residual blocks) |
| Model width | 512 |
| Attention heads | 8 |
| FFN width | 2048 (GELU) |
| Vocabulary | 5,000 BPE tokens |
| Context length | 96 tokens at training time (1,024 max) |
| Optimizer | Adam, lr 3e-4, 500 warmup steps, grad clip 5.0 |

## Build and run

Requires CMake ≥ 3.16 and a C++17 compiler. On Linux, install OpenBLAS first (`sudo apt-get install libopenblas-dev`); macOS uses the built-in Accelerate framework.

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j
```

Three modes:

```bash
# Quick end-to-end smoke test: trains a tiny model for 50 steps (~1 min)
./build/transformer train-fast

# Full training run on Tiny Shakespeare (produces shakespeare_final.bin)
./build/transformer train

# Sample from a trained checkpoint
./build/transformer generate shakespeare_final.bin "ROMEO:"

# Interactive REPL: type a prompt, watch it stream a continuation
./build/transformer chat shakespeare_final.bin

# Reproduce the BENCHMARKS.md numbers
./build/transformer bench

# Pre-tokenize a corpus for fast, memory-mapped training (see below)
./build/transformer prepare my_corpus.txt 5000
```

The first `train` run also trains the BPE tokenizer and caches it (`tokenizer_5000.cache`); later runs reuse the cache. Checkpoints are plain binary dumps of the weights plus hyperparameters, so `generate` can reconstruct the model from the file alone.

## Training on your own corpus

Any plain-text file works. For anything beyond toy size, pre-tokenize it once:

```bash
./build/transformer prepare my_corpus.txt 5000   # writes my_corpus.txt.5000.{train,val}.bin
./build/transformer train my_corpus.txt          # memory-maps the .bin files
```

`prepare` trains a BPE tokenizer on the corpus and writes the encoded tokens as binary files (uint16 per token, 95/5 train/val split). Training memory-maps them, so the corpus is never re-encoded and usable corpus size is bounded by disk, not RAM — the kernel pages in only the windows each batch actually touches. Without the `.bin` files, `train` falls back to encoding the corpus in memory, which is fine at Tiny Shakespeare scale.

Performance across optimization iterations is tracked in [BENCHMARKS.md](BENCHMARKS.md) (`./build/transformer bench` reproduces the numbers).

## Tests

```bash
ctest --test-dir build --output-on-failure
```

The test suite checks the parts that are easiest to get silently wrong:

- **Gradient checking** — analytical gradients from the autograd engine compared against central-difference numerical gradients, for individual ops through full attention blocks
- **Attention bias, weight tying, dropout** — verification of specific architectural behaviors
- `tests/integration/sanity_tests.cpp` — end-to-end training runs that must show decreasing loss

CI runs the full suite plus a training smoke test on macOS and Linux.

## Design notes

- **Explicitness over abstraction.** Every forward and backward pass is readable C++ — no expression templates, no code generation. The autograd graph is a DAG of `Variable` nodes holding closures for their backward functions; `backward()` topologically sorts and walks it.
- **Numerics matter.** Softmax and log-softmax use the max-subtraction trick; the loss path computes log-softmax + NLL rather than softmax + log; gradient checks catch regressions.
- **Performance where it counts.** Profiling showed matmul dominating, so it delegates to BLAS (`blas_wrapper.h`); everything else stays simple. The BPE tokenizer caches merges to make encoding runs fast.
- **The training loop is honest.** Loss decreases because the math is right, not because a framework fixed it — a full training run on an M2 Pro takes hours, and produced the sample above.

## Repository layout

```
include/, src/
  transformer/   tensor, variable (autograd), attention, layer_norm,
                 feedforward, embeddings, transformer_block, gpt_model,
                 optimizer, text_gen
  tokenizer/     BPE tokenizer
  data/          dataset + batching dataloader
  training/      trainer (loop, checkpointing)
  utils/         metrics, training helpers
tests/
  unit/          gradient checks and component tests
  integration/   end-to-end sanity tests
data/            Tiny Shakespeare corpus (~1.1MB)
```

## Limitations and roadmap

- A Metal (MPS) backend routes matmuls above ~10 GFLOPs to the GPU via zero-copy unified memory. At the current 22M-param scale that threshold is never crossed — measurement showed Apple's AMX (CPU) winning below it (see BENCHMARKS.md #7) — but it engages automatically at larger model/batch/context sizes. A CUDA port was attempted earlier and rolled back (see git history).
- Educational scale: don't expect it to replace your favorite inference engine.

## License

MIT
