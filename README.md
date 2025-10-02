# Transformer From Scratch (C++17)

A minimal yet complete Transformer/GPT-style model implemented entirely from scratch in modern C++ (C++17). No deep learning frameworks or external numerical libraries—just raw tensors, autograd, attention, optimizer, and a tiny tokenizer to demonstrate end-to-end training and inference mechanics.

This project is designed to highlight systems-level understanding: memory layout, numerics, backprop, and training loops, all in clean, readable C++.


## Highlights
- **From-scratch tensors and autograd**: `Tensor`, `Variable`, and reverse-mode autodiff with gradient clipping.
- **Transformer components**: `MultiHeadAttention`, `FeedForward`, `LayerNorm`, `TokenEmbedding`, `PositionalEncoding`, and `TransformerBlock` assembled into a `GPTModel`.
- **Training utilities**: Simple `Dataset` and `DataLoader` for batching synthetic sequences.
- **Tokenizer**: A small `BPE` tokenizer implementation (header and source) to illustrate tokenization infrastructure.
- **Zero third‑party dependencies**: Pure C++17, portable, and easy to read.


## Repository Structure
```
include/
  data/
    dataloader.h
    dataset.h
  tokenizer/
    bpe_tokenizer.h
  transformer/
    activations.h
    attention.h
    feedforward.h
    gpt_model.h
    layer_norm.h
    linear.h
    module.h
    multihead_attention.h
    optimizer.h
    positional_encoding.h
    tensor.h
    text_gen.h
    token_embedding.h
    transformer_block.h
    variable.h
src/
  data/
    dataloader.cpp
    dataset.cpp
  tokenizer/
    bpe_tokenizer.cpp
  transformer/
    activations.cpp
    attention.cpp
    feedforward.cpp
    gpt_model.cpp
    layer_norm.cpp
    linear.cpp
    multihead_attention.cpp
    optimizer.cpp
    positional_encoding.cpp
    tensor.cpp
    text_gen.cpp
    token_embedding.cpp
    transformer_block.cpp
    variable.cpp
  main.cpp
CMakeLists.txt
```


## What the Demo Does
The entry point `src/main.cpp` runs two self-contained trainings to verify correctness:

- **Overfitting test** (`train_overfitting_test()`):
  - Teaches a tiny GPT model to memorize a short sequence `[1,2,3,4,5]`.
  - Prints loss and gradient norms to validate gradient flow and learning.
  - Reports final token predictions and accuracy.

- **Mini-batch training with DataLoader** (`train_with_dataloader()`):
  - Trains on a synthetic token stream with batching via `DataLoader`.
  - Logs per-batch loss and average epoch loss, demonstrating stable multi-batch training.

Hyperparameters in these demos are set directly inside `src/main.cpp` for clarity and easy experimentation.


## Build and Run
Prerequisites: **CMake ≥ 3.16** and a **C++17** compiler (e.g., clang or gcc). Tested on macOS.

```bash
# Configure and build
cmake -S . -B build
cmake --build build -j

# Run (single-config generators typically place the binary directly in build/)
./build/transformer

# If you are using a multi-config generator (Xcode, MSVC), the binary may be under a config dir, e.g.:
# ./build/Debug/transformer
```

Expected console output includes clear progress for both training phases, example losses, gradient norms, and a final success/partial/failure message for the overfitting test.


## Key Files to Explore
- **`src/main.cpp`**: Training loops and demonstration harness.
- **`include/transformer/gpt_model.h`**: Model composition of blocks/attention/embeddings.
- **`include/transformer/multihead_attention.h`**: Scaled dot-product attention across heads.
- **`include/transformer/variable.h` / `include/transformer/tensor.h`**: Autograd and tensor core.
- **`include/transformer/optimizer.h`**: Adam with gradient clipping.
- **`include/data/*`**: Minimal dataset/dataloader abstractions.
- **`include/tokenizer/bpe_tokenizer.h`**: Tiny BPE tokenizer (not required for the demos but illustrates extensibility).


## Design Notes
- **Simplicity first**: The code favors readability and pedagogy over micro-optimizations.
- **Deterministic flow**: No external randomness beyond basic initialization; easy to trace.
- **No hidden magic**: Every operation—forward, backward, parameter update—is explicit and inspectable.


## Possible Extensions
- Hook up the BPE tokenizer and train on real text.
- Add dropout, weight decay, and learning rate schedulers.
- Introduce mixed precision or vectorized kernels for speed.
- Save/load checkpoints and export weights.


## Limitations
- CPU-only, educational scale; not optimized for large datasets or long contexts.
- Demos use synthetic data with hyperparameters in code for clarity.


## How This Project Demonstrates Skill
- Systems-level ML engineering in C++: building tensors/autograd/optimizers from first principles.
- Clear software structure with modular headers and source files.
- Reproducible training demos that validate correctness without external dependencies.


## License
No license specified. If you intend to use this code beyond evaluation, please add an appropriate license file.
