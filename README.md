# grad.cpp

[![CI](https://github.com/liamford1/grad.cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/liamford1/grad.cpp/actions/workflows/ci.yml)

A from-scratch autograd engine and the GPT-style language models it trains, implemented in C++17 — tensors, reverse-mode automatic differentiation with hand-derived backward passes, multi-head attention, AdamW, and a BPE tokenizer, with no ML frameworks. The only external dependency is a BLAS library (Apple Accelerate on macOS, OpenBLAS on Linux) for fast matrix multiplication. *(Formerly `transformer-from-scratch`.)*

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

About 8,300 lines of implementation and 1,500 lines of tests.

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
| Optimizer | AdamW, lr 3e-4, 500 warmup steps, grad clip 5.0 |

## Build and run

Requires CMake ≥ 3.16 and a C++17 compiler. On Linux, install OpenBLAS first (`sudo apt-get install libopenblas-dev`); macOS uses the built-in Accelerate framework.

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j
```

Common commands:

```bash
# Quick end-to-end smoke test: trains a tiny model for 50 steps (~1 min)
./build/grad train-fast

# Full training run on Tiny Shakespeare (produces shakespeare_final.bin)
./build/grad train

# Sample from a trained checkpoint
./build/grad generate shakespeare_final.bin "ROMEO:"

# Interactive REPL: type a prompt, watch it stream a continuation
./build/grad chat shakespeare_final.bin

# Reproduce the benchmark with repeated trials and a machine-readable record
./build/grad bench --steps 20 --trials 5 --json grad-benchmark.json

# Pre-tokenize a corpus for fast, memory-mapped training (see below)
./build/grad prepare my_corpus.txt 5000
```

The first `train` run also trains the BPE tokenizer and caches it (`tokenizer_5000.cache`); later runs reuse the cache. Checkpoints are plain binary dumps of the weights plus hyperparameters, so `generate` can reconstruct the model from the file alone.

The build also produces an installable `grad::core` CMake target:

```bash
cmake --install build --prefix ./dist
```

Downstream CMake projects can use `find_package(grad CONFIG REQUIRED)` and link `grad::core` after adding `dist` to `CMAKE_PREFIX_PATH`.

Release builds are tuned for the build machine with `-march=native` (that is how every number in [BENCHMARKS.md](BENCHMARKS.md) was measured). For a binary you intend to run on another CPU, configure with `-DGRAD_NATIVE_ARCH=OFF`.

## Training on your own corpus

Any plain-text file works. For anything beyond toy size, pre-tokenize it once:

```bash
./build/grad prepare my_corpus.txt 5000   # writes my_corpus.txt.5000.{train,val}.bin
./build/grad train my_corpus.txt          # memory-maps the .bin files
```

`prepare` trains a BPE tokenizer on the corpus (sampling the first 32MB for merge learning on large corpora — frequencies converge long before that) and writes the encoded tokens as binary files (uint16 per token, 95/5 train/val split). Training memory-maps them, so the corpus is never re-encoded and usable corpus size is bounded by disk, not RAM — the kernel pages in only the windows each batch actually touches. Without the `.bin` files, `train` falls back to encoding the corpus in memory, which is fine at Tiny Shakespeare scale.

Three model presets are built in:

| preset | params | config | intended for |
|---|---|---|---|
| `small` (default) | ~22M | d512 × 6L × 8H, seq 96, batch 8, vocab 5k | Tiny Shakespeare |
| `medium` | ~70M | d768 × 8L × 12H, seq 256, batch 8×4 accum, vocab 16k | TinyStories-scale corpora |
| `modern` | ~70M | `medium` with a Llama-style block: RMSNorm, RoPE, bias-free SwiGLU | architecture A/B against `medium` |

`modern` swaps the GPT-2-style block (LayerNorm, learned positional embeddings, GELU FFN) for the recipe used by current LLMs, at the same parameter count and training budget — so a `medium` run and a `modern` run on the same corpus A/B the architecture and nothing else. Checkpoints record their architecture (format v2; older files load as GPT-2-style) and `modern` checkpoints are prefixed `<corpus>_modern_*` so both lineages coexist.

`medium`'s logits matmul crosses the ~10 GFLOP threshold where the Metal GPU backend engages (BENCHMARKS.md #7); its micro-batch size is set by memory, not compute — one micro-batch peaks at ~4GB of footprint, sized to fit a 16GB machine (BENCHMARKS.md #8–9) — and gradient accumulation gives the optimizer an effective batch of 32 at that same peak. Example end-to-end:

```bash
curl -L -o data/tinystories.txt \
  "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
./build/grad prepare data/tinystories.txt 16000
./build/grad train data/tinystories.txt medium     # writes tinystories_final.bin
./build/grad chat tinystories_final.bin data/tinystories.txt 16000
```

Every run logs per-step metrics to `<prefix>_metrics.csv`, and a live terminal dashboard renders them — loss curves on a braille canvas (raw + EMA), the validation track with running best, gradient-norm and step-time sparklines, progress and ETA. Open it in a second terminal while training:

```bash
./build/grad watch                      # newest run in this directory
./build/grad watch tinystories_modern   # or a specific run prefix
```

It refreshes once a second, works on finished runs too (the CSV is the run's permanent record), and `q` quits.

Long runs are interruptible: Ctrl-C saves a resume pair (`<prefix>_resume_model.bin` + `<prefix>_resume_state.bin` — weights, Adam moments, and schedule position), which is also refreshed at every eval interval, so a crash costs at most a few minutes of work.

```bash
./build/grad train data/tinystories.txt medium resume               # continue where it stopped
./build/grad train data/tinystories.txt medium tinystories_best.bin # warm-start: weights only, fresh schedule
```

A resumed or warm-started run reseeds the data loader (by step position and checkpoint path respectively), so it draws fresh training windows instead of replaying the batches the checkpoint already saw.

## Performance

Training throughput for the 22M benchmark config vs PyTorch 2.13 on the same M2 Pro (fp32). The PyTorch side ([`benchmarks/pytorch_baseline.py`](benchmarks/pytorch_baseline.py)) builds the identical model with idiomatic fused QKV and `scaled_dot_product_attention`:

| training config | grad.cpp (CPU) | PyTorch (CPU) | PyTorch (MPS GPU) |
|---|---:|---:|---:|
| 22M · d512 L6 · seq 96 | **5,418 tok/s** | 3,685 | 8,112 |

On this specific CPU workload, grad.cpp is 1.5× faster than PyTorch; PyTorch MPS is 1.5× faster than grad.cpp. This is a specialized workload result, not a claim of general framework superiority. The 70M comparison previously shown here was withdrawn after a source-metrics audit found an inconsistent throughput calculation; [BENCHMARKS.md](BENCHMARKS.md) records the correction.

The full optimization history — 1.2 → 7.9 steps/s across 11 measured rounds, including null results — is in [BENCHMARKS.md](BENCHMARKS.md). Current benchmark commands run repeated trials, report the median, identify dirty builds, record the compiler/system/backend, and optionally write JSON.

## Tests

```bash
ctest --test-dir build --output-on-failure
```

The test suite checks the parts that are easiest to get silently wrong:

- **Gradient checking** — analytical gradients from the autograd engine compared against central-difference numerical gradients, for individual ops through full attention blocks
- **Attention bias, weight tying, dropout** — verification of specific architectural behaviors
- **Integration checks** — tiny-sequence overfitting plus parity between full-sequence and KV-cached inference for both architectures
- **Hardware-aware results** — Metal parity is reported as skipped, not passed, when no Metal device is exposed

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
